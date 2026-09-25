"""
Tests for the three tools, against the real dataset. None of these
need an LLM - the tools are pure functions the orchestrator calls
after the model has already decided which one to use.
"""
import json
import math

import pytest

from app.data.loader import load_dataset
from app.tools.booking import book_viewing
from app.tools.leads import capture_lead
from app.tools.search import search_inventory


@pytest.fixture(scope="module")
def df():
    return load_dataset()


# --- search_inventory -------------------------------------------------

def test_structured_filter_matches_only_correct_rows(df):
    result = search_inventory(df, None, make="Mercedes-Benz", year_min=2023)
    assert result["total_matched"] > 0
    assert all(r["make"] == "mercedes-benz" and r["year"] >= 2023 for r in result["results"])


def test_absent_make_returns_zero_matches_not_invented_cars(df):
    # Porsche is confirmed absent from this dataset (Tesla, unexpectedly, IS present)
    result = search_inventory(df, None, make="Porsche")
    assert result["total_matched"] == 0
    assert result["results"] == []


def test_price_filter_excludes_and_notes_unconfirmed_prices(df):
    result = search_inventory(df, None, price_max=50000)
    assert all(r["price_extracted"] <= 50000 for r in result["results"])
    assert result["note"] is not None and "no confirmed price" in result["note"]


def test_results_are_json_safe_even_with_missing_price(df):
    # Regression test for a real bug: raw NaN in price_extracted crashed
    # the *next* LLM call's JSON encoding downstream in the orchestrator.
    result = search_inventory(df, None, make="Toyota")
    for row in result["results"]:
        price = row["price_extracted"]
        assert price is None or not (isinstance(price, float) and math.isnan(price))
    json.dumps(result, allow_nan=False)  # must not raise


# --- book_viewing -------------------------------------------------

@pytest.mark.parametrize("day,time,expected", [
    ("Tuesday", "10:00 AM", True),
    ("Sunday", "10:00 AM", False),   # closed Sundays
    ("Monday", "9:00 PM", False),    # after hours
    ("Wednesday", "8:00 PM", True),  # boundary - last valid slot
    ("Thursday", "2pm", True),       # informal time format
])
def test_book_viewing_day_and_hour_rules(day, time, expected):
    assert book_viewing(day, time)["success"] is expected


def test_book_viewing_rejects_unparseable_time():
    assert book_viewing("Friday", "not a time")["success"] is False


# --- capture_lead -------------------------------------------------

def test_capture_lead_writes_header_once_and_appends(tmp_path):
    leads_csv = tmp_path / "leads.csv"
    capture_lead(leads_csv, user_id="user-1", name="Ayesha", price_min=40000, price_max=60000, needs="family SUV")
    capture_lead(leads_csv, user_id="user-2", needs="cheap first car")

    content = leads_csv.read_text()
    assert content.count("timestamp,user_id") == 1
    assert "Ayesha" in content
    assert "user-2" in content
