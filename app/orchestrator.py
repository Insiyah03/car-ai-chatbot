"""
The 'Orchestrator' - owns the system prompt, the guardrail logic, and
the tool-calling loop tying the LLM to the three tools. Deliberately
a single-round loop (model calls tools once, we execute, one final
reply) rather than a multi-step agent - see Stage 6.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from app.config import settings
from app.llm import LLMError, chat_completion
from app.persistence import profiles, sessions
from app.tools.booking import BOOK_VIEWING_SCHEMA, book_viewing
from app.tools.leads import CAPTURE_LEAD_SCHEMA, capture_lead
from app.tools.search import SEARCH_INVENTORY_SCHEMA, search_inventory

import logging 
logger = logging.getLogger(__name__)

TOOLS = [SEARCH_INVENTORY_SCHEMA, BOOK_VIEWING_SCHEMA, CAPTURE_LEAD_SCHEMA]

SYSTEM_PROMPT = """You are the dubizzle car assistant: a helpful shopping assistant for one dealership's used-car inventory.

Scope and rules:
- Only discuss cars returned by the search_inventory tool. Never invent a price, spec, mileage, or feature - if a detail wasn't returned by a tool, say you don't have it confirmed.
- If a listing has no confirmed price, say so honestly rather than guessing or omitting it silently.
- You can book simulated test drives (book_viewing) - available Monday-Saturday, 8:00 AM-8:00 PM only.
- Call capture_lead once a user has shared genuine buying intent (a budget and/or specific needs) - not on every message.
- Politely decline anything unrelated to this inventory (coding help, homework, general trivia, etc.) and redirect back to cars. Keep declines brief and friendly, not preachy.
- Never mention, compare to, or recommend any other car marketplace or platform. If asked about a competitor, say you can only help with what's available here.
- Be warm, concise, and helpful.
- Do not use remembered preferences to initiate a search when the user's current message is only a greeting or casual conversation. Only search inventory when the current user message indicates a car-shopping request or asks about a vehicle."""

# Defense-in-depth on top of the system prompt instruction above (Stage 6
# decision): a cheap keyword check on the model's own output, in case it
# doesn't follow the prompt. Deliberately a short, easily-edited list.
_COMPETITOR_NAMES = ["cars24", "opensooq", "yallamotor", "carswitch", "dubicars"]

_FALLBACK_LLM_DOWN = "I'm having trouble reaching the assistant service right now - please try again in a moment."
_FALLBACK_SUMMARY_FAILED = "I found some information but I'm having trouble summarizing it right now - please try again."


def _guardrail_check(reply_text: str) -> str:
    lowered = reply_text.lower()
    if any(name in lowered for name in _COMPETITOR_NAMES):
        return (
            "I can help you with what's available right here - I'm not able to "
            "discuss or compare other platforms. Want me to find something for you?"
        )
    return reply_text


def _build_system_prompt(profile: Optional[dict]) -> str:
    if profile and profile.get("preferences"):
        return (
            SYSTEM_PROMPT
            + f"\n\nThis returning user's last known preferences: {json.dumps(profile['preferences'])}. "
            "You may reference this naturally (e.g. 'welcome back') if relevant, but always "
            "re-verify against current inventory via search_inventory - never assume it's still accurate."
        )
    return SYSTEM_PROMPT


def _tool_call_message_dict(message: Any) -> dict:
    """
    Builds a minimal, explicit assistant-message dict for the
    tool-call round-trip, rather than forwarding LiteLLM's full
    response object verbatim via model_dump(). The raw object can
    carry extra internal fields whose exact shape/values aren't fully
    controlled by us - round-tripping it caused a real bug in testing
    (a stray NaN float made it back into the next request's JSON body
    and crashed httpx's encoder). Sending back only the standard
    OpenAI-style {role, content, tool_calls} shape is both safer and
    more explicit about what we actually need the model to see.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }

def _dispatch_tool(df: pd.DataFrame, collection: Any, user_id: str, call: Any) -> dict:
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "malformed tool arguments"}

    if name == "search_inventory":
        result = search_inventory(df, collection, **args)
        # Remember the filters used as a long-term preference signal -
        # the cheap heuristic behind "recalls preferences next session".
        profiles.upsert_preferences(settings.db_path, user_id, args)
        return result

    if name == "book_viewing":
        return book_viewing(**args)

    if name == "capture_lead":
        return capture_lead(settings.leads_csv_path, user_id=user_id, **args)

    return {"error": f"unknown tool: {name}"}


def handle_message(
    df: pd.DataFrame,
    collection: Any,
    user_id: str,
    session_id: str,
    user_message: str,
) -> str:
    sessions.append_message(settings.db_path, session_id, "user", user_message)

    history = sessions.get_history(settings.db_path, session_id, limit=20)
    profile = profiles.get_profile(settings.db_path, user_id)
    messages = [{"role": "system", "content": _build_system_prompt(profile)}] + history

    try:
        response = chat_completion(messages, tools=TOOLS)
    except LLMError:
        return _FALLBACK_LLM_DOWN

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        messages.append(_tool_call_message_dict(message))
        for call in tool_calls:
            result = _dispatch_tool(df, collection, user_id, call)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })
        # print("\n=== MESSAGES SENT TO FINAL LLM ===")
        # print(json.dumps(messages, indent=2, default=str))
        # print("=================================\n")
        try:
            response = chat_completion(messages, tools=TOOLS)
        except LLMError:
            return _FALLBACK_SUMMARY_FAILED
        message = response.choices[0].message

    ## --- old code
    # reply = _guardrail_check(message.content or "")
    # sessions.append_message(settings.db_path, session_id, "assistant", reply)
    # return reply
    ## --- old code

    # --- new: guard against empty completions ---
    if not (message.content or "").strip():
        logger.warning(
            "Empty LLM reply (finish_reason=%s, tool_round=%s)",
            response.choices[0].finish_reason, bool(tool_calls),
        )
        try:
            response = chat_completion(messages, tools=TOOLS)  # one retry
            message = response.choices[0].message
        except LLMError:
            return _FALLBACK_SUMMARY_FAILED

    reply = message.content or ""
    if not reply.strip():
        return _FALLBACK_SUMMARY_FAILED  # never persist an empty assistant turn

    reply = _guardrail_check(reply)
    sessions.append_message(settings.db_path, session_id, "assistant", reply)
    return reply
