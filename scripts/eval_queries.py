"""
A small, deliberately lightweight evaluation harness (Stage 2's
"Should Have" small eval set). Hits the LIVE, running backend with a
battery of canned conversations and prints the results for review.

This is NOT an auto-grading LLM-as-judge harness - building one would
be over-engineering for a prototype at this scale (see README). Where
a check can be done cheaply and reliably (e.g. "did the reply mention
a competitor"), it's automated below. Everything else is printed
clearly against a "what to check" note for a human to review - that
level of rigor matches what the brief actually asks for without
inventing complexity nothing here justifies.

Requires the backend running first:  uv run uvicorn main:app --reload
Then, in another terminal:           uv run python scripts/eval_queries.py
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@dataclass
class Scenario:
    name: str
    turns: list[str]
    what_to_check: str
    auto_check: Optional[Callable[[str], tuple[bool, str]]] = None


def _no_competitor_mentioned(reply: str) -> tuple[bool, str]:
    competitors = ["cars24", "opensooq", "yallamotor", "carswitch", "dubicars"]
    hit = next((c for c in competitors if c in reply.lower()), None)
    return (False, f"mentioned competitor '{hit}'") if hit else (True, "no competitor mentioned")


def _mentions_no_matches(reply: str) -> tuple[bool, str]:
    signals = ["don't have", "no matches", "not available", "couldn't find", "no listing", "not in our inventory", "no porsche"]
    if any(s in reply.lower() for s in signals):
        return True, "correctly indicated no matches"
    return False, "did not clearly indicate no matches - check for a hallucinated car"


SCENARIOS = [
    Scenario(
        name="Structured search",
        turns=["Show me Mercedes-Benz cars from 2023 or newer"],
        what_to_check="Every car mentioned should actually be a Mercedes-Benz, 2023+.",
    ),
    Scenario(
        name="Fuzzy / semantic search",
        turns=["I need something family-friendly and good for road trips"],
        what_to_check="Suggestions should be plausible family/road-trip vehicles (SUVs, larger sedans), not arbitrary.",
    ),
    Scenario(
        name="Price honesty",
        turns=["What cars do you have under 50,000 AED?"],
        what_to_check="Only cars with a CONFIRMED price under 50k should be listed as matching - should not silently claim a price for a listing with none confirmed.",
    ),
    Scenario(
        name="Grounding - absent make",
        turns=["Do you have any Porsches?"],
        what_to_check="Must say no / not available - Porsche is confirmed absent from this dataset.",
        auto_check=_mentions_no_matches,
    ),
    Scenario(
        name="Multi-turn coreference",
        turns=["Show me Toyota cars", "What's the year on the first one?"],
        what_to_check="Second reply should correctly refer back to a specific car from the first reply, not lose context or ask 'which car?'.",
    ),
    Scenario(
        name="Booking - valid slot",
        turns=["Book me a test drive for Tuesday at 10am"],
        what_to_check="Should confirm the booking - Tuesday 10am is within Mon-Sat, 8am-8pm.",
    ),
    Scenario(
        name="Booking - invalid slot",
        turns=["Book me a test drive for Sunday at 3pm"],
        what_to_check="Should decline and explain we're closed Sundays - not silently accept it.",
    ),
    Scenario(
        name="Lead capture",
        turns=["I'm looking for something between 40,000 and 60,000 AED, ideally an SUV. My name is Ayesha."],
        what_to_check="Should recognize this as buying intent and capture a lead - check data/leads.csv for a new row after running this.",
    ),
    Scenario(
        name="Guardrail - off-topic",
        turns=["Can you write me a Python script to sort a list?"],
        what_to_check="Should politely decline and redirect to cars, not answer the coding question.",
    ),
    Scenario(
        name="Guardrail - competitor",
        turns=["What do you think of Cars24? Is it better than this?"],
        what_to_check="Should not compare to or recommend a competitor platform.",
        auto_check=_no_competitor_mentioned,
    ),
]


def run_scenario(scenario: Scenario) -> None:
    print(f"\n{'=' * 70}\n{scenario.name}\n{'=' * 70}")
    print(f"What to check: {scenario.what_to_check}")

    user_id = f"eval-{uuid.uuid4().hex[:8]}"
    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    final_reply = ""

    for turn in scenario.turns:
        print(f"\n> {turn}")
        try:
            response = httpx.post(
                f"{BACKEND_URL}/chat",
                json={"user_id": user_id, "session_id": session_id, "message": turn},
                timeout=30.0,
            )
            response.raise_for_status()
            final_reply = response.json()["reply"]
        except httpx.HTTPError as exc:
            final_reply = f"[REQUEST FAILED: {exc}]"
        print(f"< {final_reply}")

    if scenario.auto_check:
        passed, reason = scenario.auto_check(final_reply)
        print(f"\n[auto-check: {'PASS' if passed else 'FAIL'}] {reason}")


def main() -> None:
    print(f"Running {len(SCENARIOS)} eval scenarios against {BACKEND_URL}")
    print("Make sure the backend is running: uv run uvicorn main:app --reload\n")
    for scenario in SCENARIOS:
        run_scenario(scenario)
    print(f"\n{'=' * 70}")
    print("Done. Scenarios without an [auto-check] line need manual review")
    print("against their 'What to check' note above.")


if __name__ == "__main__":
    main()
