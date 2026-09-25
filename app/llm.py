"""
Thin wrapper around LiteLLM's completion() and embedding() calls.
Owns retry/backoff for the free-tier Gemini key (via app.retry) so
the orchestrator and vector store never deal with raw API failures
directly - just an LLMError they can catch and turn into a graceful
fallback reply.
"""
from __future__ import annotations

from typing import Any, Optional

import litellm

from app.config import settings
from app.retry import RetriesExhausted, with_retries

# All of these inherit from the matching openai.* exception type
# (docs.litellm.ai/docs/exception_mapping), which is what makes this
# work uniformly even if we ever swap providers. Deliberately excludes
# AuthenticationError/BadRequestError - those are OUR bugs (bad key,
# malformed request), not transient failures, and should fail fast
# and loud rather than silently retry and hide the real problem.
_RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


class LLMError(Exception):
    """Raised when an LLM call ultimately fails after retries."""


def chat_completion(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    max_retries: int = 2,
) -> Any:
    """Returns the raw LiteLLM ModelResponse (message + optional tool_calls)."""
    def _call():
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "api_key": settings.require_api_key(),
            "timeout": 20,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return litellm.completion(**kwargs)

    try:
        return with_retries(_call, _RETRYABLE_EXCEPTIONS, max_retries=max_retries)
    except RetriesExhausted as exc:
        raise LLMError("chat completion failed after retries") from exc


# def embed_texts(texts: list[str], max_retries: int = 2) -> list[list[float]]:
#     """Returns one embedding vector per input text, in the same order."""
#     def _call():
#         response = litellm.embedding(
#             model=settings.embedding_model,
#             input=texts,
#             api_key=settings.require_api_key(),
#             timeout=20,
#         )
#         return [item["embedding"] for item in response["data"]]

#     try:
#         return with_retries(_call, _RETRYABLE_EXCEPTIONS, max_retries=max_retries,sleep=3)
#     except RetriesExhausted as exc:
#         raise LLMError("embedding call failed after retries") from exc

import time

def embed_texts(
    texts: list[str],
    batch_size: int = 15,
    batch_delay: float = 2.0,
    max_retries: int = 2,
) -> list[list[float]]:
    """Returns one embedding vector per input text, in the same order."""

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        def _call():
            response = litellm.embedding(
                model=settings.embedding_model,
                input=batch,
                api_key=settings.require_api_key(),
                timeout=20,
            )
            return [item["embedding"] for item in response["data"]]

        try:
            embeddings = with_retries(
                _call,
                _RETRYABLE_EXCEPTIONS,
                max_retries=max_retries,
            )
        except RetriesExhausted as exc:
            raise LLMError("embedding call failed after retries") from exc

        all_embeddings.extend(embeddings)

        # Wait before the next batch
        if i + batch_size < len(texts):
            time.sleep(batch_delay)

    return all_embeddings