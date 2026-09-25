"""
Long-term memory CRUD: read/merge a user's stored preferences, keyed
by the client-persisted user_id (Stage 4 decision).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.persistence.db import get_connection


def get_profile(db_path: Path, user_id: str) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "user_id": row["user_id"],
        "name": row["name"],
        "preferences": json.loads(row["preferences_json"]),
        "liked_listings": json.loads(row["liked_listings_json"]),
    }


def upsert_preferences(
    db_path: Path,
    user_id: str,
    preferences: dict[str, Any],
    name: Optional[str] = None,
) -> None:
    """Merges non-null preference keys into the stored profile (creating it if new)."""
    clean_prefs = {k: v for k, v in preferences.items() if v is not None}
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT preferences_json FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existing:
            merged = json.loads(existing["preferences_json"])
            merged.update(clean_prefs)
            conn.execute(
                "UPDATE user_profile SET preferences_json = ?, "
                "name = COALESCE(?, name), updated_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ?",
                (json.dumps(merged), name, user_id),
            )
        else:
            conn.execute(
                "INSERT INTO user_profile (user_id, name, preferences_json) VALUES (?, ?, ?)",
                (user_id, name, json.dumps(clean_prefs)),
            )
