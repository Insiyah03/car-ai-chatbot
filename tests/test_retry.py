"""
Tests for the generic retry-with-backoff policy (app/retry.py), using
injected instant "sleep" so these run in milliseconds, not seconds.
"""
import pytest

from app.retry import RetriesExhausted, with_retries


class _FlakyError(Exception):
    pass


class _OtherError(Exception):
    pass


def test_recovers_within_retry_budget():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FlakyError("rate limited")
        return "ok"

    sleeps = []
    result = with_retries(flaky, (_FlakyError,), max_retries=2, base_delay=0.5, sleep=sleeps.append)

    assert result == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff


def test_exhausts_and_raises_after_max_retries():
    def always_fails():
        raise _FlakyError("still rate limited")

    with pytest.raises(RetriesExhausted):
        with_retries(always_fails, (_FlakyError,), max_retries=2, sleep=lambda s: None)


def test_non_retryable_exception_propagates_immediately():
    calls = {"n": 0}

    def bad_key():
        calls["n"] += 1
        raise _OtherError("invalid api key")

    with pytest.raises(_OtherError):
        with_retries(bad_key, (_FlakyError,), max_retries=2, sleep=lambda s: None)

    assert calls["n"] == 1  # never retried
