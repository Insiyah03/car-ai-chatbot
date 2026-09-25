"""
book_viewing() - simulated test-drive slot booking. Validates the
requested day/time falls Mon-Sat, 8:00 AM-8:00 PM; no real calendar
backing it (explicitly out of scope per Stage 2/5 - noted in README).
"""
from __future__ import annotations

import re
from typing import Optional

BOOK_VIEWING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "book_viewing",
        "description": "Book a simulated test-drive slot. Available Monday-Saturday, 8:00 AM-8:00 PM.",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "Day of week, e.g. 'Tuesday'"},
                "time": {"type": "string", "description": "Time, e.g. '10:00 AM' or '14:00'"},
                "listing_id": {"type": "integer", "description": "The listing_id being viewed, if known"},
                "listing_title": {"type": "string", "description": "Human-readable car name, for the confirmation message"},
            },
            "required": ["day", "time"],
        },
    },
}

_VALID_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday"}
_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE)


def _parse_time(time_str: str) -> tuple[int, int]:
    match = _TIME_RE.match(time_str.strip())
    if not match:
        raise ValueError(f"unrecognized time format: {time_str!r}")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"time out of range: {time_str!r}")
    return hour, minute


def book_viewing(
    day: str,
    time: str,
    listing_id: Optional[int] = None,
    listing_title: Optional[str] = None,
) -> dict:
    if day.strip().lower() not in _VALID_DAYS:
        return {
            "success": False,
            "reason": f"Test drives are only available Monday-Saturday - '{day}' isn't bookable (we're closed Sundays).",
        }

    try:
        hour, minute = _parse_time(time)
    except ValueError:
        return {
            "success": False,
            "reason": f"I couldn't understand the time '{time}'. Try a format like '10:00 AM'.",
        }

    within_hours = (8 <= hour < 20) or (hour == 20 and minute == 0)
    if not within_hours:
        return {
            "success": False,
            "reason": "Test drives are available between 8:00 AM and 8:00 PM.",
        }

    confirmation = f"Test drive booked for {day.title()} at {time}"
    if listing_title:
        confirmation += f" - {listing_title}"
    return {
        "success": True,
        "confirmation": confirmation,
        "day": day.title(),
        "time": time,
        "listing_id": listing_id,
    }
