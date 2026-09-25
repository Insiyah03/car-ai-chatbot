"""
Generic retry-with-backoff helper, deliberately free of any LLM/HTTP
dependency so it can be unit-tested in isolation. app/llm.py supplies
the litellm-specific exception types; this module just implements
the retry policy once so it isn't duplicated for completion() and
embedding() separately.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetriesExhausted(Exception):
    """Raised when fn() still fails after all retry attempts."""


def with_retries(
    fn: Callable[[], T],
    retryable_exceptions: tuple[type[Exception], ...],
    max_retries: int = 2,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "call failed (%s: %s), retrying in %.1fs (attempt %d/%d)",
                type(exc).__name__, exc, delay, attempt + 1, max_retries,
            )
            sleep(delay)
    raise RetriesExhausted(f"failed after {max_retries + 1} attempts") from last_exc
