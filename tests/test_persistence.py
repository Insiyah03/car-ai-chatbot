"""
Tests for the unified SQLite persistence layer: short-term (sessions)
and long-term (user_profile) memory, including the cross-session
preference-merge behavior the "long-term memory" grading criterion
depends on.
"""
import pytest

from app.persistence import profiles, sessions
from app.persistence.db import init_db


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_session_history_returns_most_recent_n_in_order(db_path):
    for i in range(5):
        sessions.append_message(db_path, "sess-A", "user", f"msg{i}")
        sessions.append_message(db_path, "sess-A", "assistant", f"reply{i}")

    history = sessions.get_history(db_path, "sess-A", limit=4)
    assert [h["content"] for h in history] == ["msg3", "reply3", "msg4", "reply4"]


def test_sessions_are_isolated_by_session_id(db_path):
    sessions.append_message(db_path, "sess-A", "user", "hello")
    assert sessions.get_history(db_path, "sess-B") == []


def test_long_term_preferences_merge_across_simulated_sessions(db_path):
    # Session 1: first visit
    profiles.upsert_preferences(db_path, "user-1", {"make": "Toyota", "price_max": 60000}, name="Ayesha")
    p1 = profiles.get_profile(db_path, "user-1")
    assert p1["name"] == "Ayesha"
    assert p1["preferences"] == {"make": "Toyota", "price_max": 60000}

    # Session 2: user returns later, searches again with an updated filter
    profiles.upsert_preferences(db_path, "user-1", {"price_max": 45000, "keywords": "family SUV"})
    p2 = profiles.get_profile(db_path, "user-1")
    assert p2["name"] == "Ayesha"  # preserved via COALESCE, not overwritten with None
    assert p2["preferences"] == {"make": "Toyota", "price_max": 45000, "keywords": "family SUV"}


def test_unknown_user_returns_none(db_path):
    assert profiles.get_profile(db_path, "nobody") is None
