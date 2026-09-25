"""
Short-term memory CRUD: append/read the message history for a
session_id, from the shared SQLite engine (Stage 4 decision).
"""
from __future__ import annotations

from pathlib import Path

from app.persistence.db import get_connection


def append_message(db_path: Path, session_id: str, role: str, content: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_history(db_path: Path, session_id: str, limit: int = 20) -> list[dict]:
    """Returns the most recent `limit` messages, oldest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM sessions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
