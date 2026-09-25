"""
capture_lead() - writes qualified lead info to data/leads.csv,
simulating lead recording per the brief's explicit requirement
(a separate artifact from SQLite persistence - see Stage 4).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CAPTURE_LEAD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "capture_lead",
        "description": (
            "Record this user as a qualified lead once they've shared real "
            "buying intent (a price range and/or specific needs). Call this "
            "at most once per clear intent signal, not on every message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The user's name, if given"},
                "price_min": {"type": "number", "description": "Minimum budget in AED, if given"},
                "price_max": {"type": "number", "description": "Maximum budget in AED, if given"},
                "needs": {"type": "string", "description": "Free-text summary of what they're looking for"},
                "listing_id_interested": {"type": "integer", "description": "A specific listing they're interested in, if any"},
            },
            "required": [],
        },
    },
}

_FIELDS = ["timestamp", "user_id", "name", "price_range", "needs", "listing_id_interested"]


def capture_lead(
    leads_csv_path: Path,
    user_id: str,
    name: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    needs: Optional[str] = None,
    listing_id_interested: Optional[int] = None,
) -> dict:
    leads_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = leads_csv_path.exists()

    price_range = ""
    if price_min is not None or price_max is not None:
        price_range = f"{price_min if price_min is not None else ''}-{price_max if price_max is not None else ''}"

    with open(leads_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "name": name or "",
            "price_range": price_range,
            "needs": needs or "",
            "listing_id_interested": listing_id_interested if listing_id_interested is not None else "",
        })

    return {"success": True, "message": "Lead recorded."}
