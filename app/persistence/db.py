"""
SQLite engine + schema for both memory tiers, per the Stage 4 decision
to unify short-term (sessions) and long-term (user_profile) state
under one persistence technology rather than splitting them across
an in-memory dict and a database.

A new connection is opened per call rather than shared across threads -
the simplest safe pattern at this scale (FastAPI's sync path handlers
run in a threadpool, and sqlite3 connections aren't thread-safe to
share). Connection pooling is explicitly a Future/Production concern
(Stage 5), not needed for a single-process prototype.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id             TEXT PRIMARY KEY,
    name                TEXT,
    preferences_json    TEXT NOT NULL DEFAULT '{}',
    liked_listings_json TEXT NOT NULL DEFAULT '[]',
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
