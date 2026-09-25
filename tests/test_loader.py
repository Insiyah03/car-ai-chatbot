"""
Tests for the dataset loader and price-extraction logic - the piece
found (during Stage 1) to need the most care, since price only exists
in messy free text. See app/data/loader.py's module docstring.
"""
from app.data.loader import extract_price, clean_html, load_dataset


def test_extracts_explicit_cash_price_as_high_confidence():
    price, confidence = extract_price("Selling for AED 115,750.00 in cash, no haggling")
    assert price == 115750.00
    assert confidence == "high"


def test_ignores_monthly_instalment_amounts():
    price, confidence = extract_price("AED 3,580 Monthly (20% Down Payment) Over 5 Years")
    assert price is None
    assert confidence == "none"


def test_ignores_pm_abbreviation_for_monthly():
    price, confidence = extract_price("Option 1 - AED 3,413 P/M for 5 years")
    assert price is None
    assert confidence == "none"


def test_ignores_unrelated_salary_or_bank_statement_amounts():
    price, confidence = extract_price("Salary requirement: AED 3000 (WPS)")
    assert price is None
    assert confidence == "none"


def test_handles_numbers_without_thousand_separators():
    price, confidence = extract_price("---   AED 106000  ------------")
    assert price == 106000.0
    assert confidence == "medium"


def test_handles_space_separated_thousands():
    price, confidence = extract_price("AED 89 900 or AED 1380/ Month with a 20% Down Payment")
    assert price == 89900.0
    assert confidence == "medium"


def test_no_price_mentioned_at_all():
    price, confidence = extract_price("Great condition, single owner, full service history.")
    assert price is None
    assert confidence == "none"


def test_clean_html_strips_tags():
    assert clean_html("Great car<br>Low mileage<br><br>Contact us") == "Great car Low mileage Contact us"


def test_load_dataset_shape():
    df = load_dataset()
    assert len(df) == 100  # padding rows dropped
    expected_columns = {
        "listing_id", "make", "model", "trim", "year", "title",
        "description_clean", "photo_url", "price_extracted", "price_confidence",
    }
    assert expected_columns.issubset(set(df.columns))
    # every row has a confidence level, and only high/medium rows have a price
    assert df["price_confidence"].isin(["high", "medium", "none"]).all()
    assert (df.loc[df["price_confidence"] == "none", "price_extracted"].isna()).all()
    assert (df.loc[df["price_confidence"] != "none", "price_extracted"].notna()).all()
