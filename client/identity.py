"""
Local, file-based "client-persisted" user ID (Stage 4 decision).
Separated from app.py so this logic is testable without Streamlit
installed/running - the UI layer should never be the only place
business-relevant logic like identity lives.

Trade-off, stated honestly (see README known limitations): this
identifies the *machine running the Streamlit client*, not a browser
or device. Same machine -> same user across restarts (what the demo
needs to prove long-term memory works). A different machine looks
like a new user - there's no real cross-device identity here, which
is fine for a local prototype but would need real auth in production.
"""
from __future__ import annotations

import uuid
from pathlib import Path

DEFAULT_ID_FILE = Path(__file__).resolve().parent.parent / "data" / ".local_user_id"


def get_or_create_user_id(id_file: Path = DEFAULT_ID_FILE) -> str:
    if id_file.exists():
        existing = id_file.read_text().strip()
        if existing:
            return existing
    new_id = str(uuid.uuid4())
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(new_id)
    return new_id
