"""
Loads data/cars_dataset.xlsx, drops the padding blank rows, and adds
derived price_extracted / price_confidence columns parsed out of the
free-text title/description - the fix for the "no structured Price
column" finding from Stage 1.

Deliberately does not import app.config, so this module (and its
price-extraction logic) can be tested and reasoned about in complete
isolation from the LLM/API-key setup.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cars_dataset.xlsx"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# handles "115,750.00", "106000", and the single-space-thousands style "89 900"
_NUMBER = r"\d[\d,]*(?:\s\d{3})?(?:\.\d{1,2})?"

_CASH_PRICE_RE = re.compile(
    rf"AED\s*({_NUMBER})\s*(?:in\s+cash|cash)", re.IGNORECASE,
)
# Anything in this window means the AED figure isn't the car's price:
# a monthly instalment (incl. "P/M", "P.M", "PM" abbreviations) or an
# unrelated amount like a salary/bank-statement financing requirement.
_MONTHLY_CONTEXT_RE = re.compile(
    r"month|down[\s-]*payment|\bp[\s./]*m\b|salary|bank\s*statement|\bwps\b",
    re.IGNORECASE,
)
_GENERIC_AED_RE = re.compile(rf"AED\s*({_NUMBER})", re.IGNORECASE)


def clean_html(raw: Optional[str]) -> str:
    """Strip <br> and other HTML tags dealer descriptions are full of."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = text.replace("&amp;", "&")
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_price(text: Optional[str]) -> tuple[Optional[float], str]:
    """
    Returns (price, confidence).
    confidence is "high" (explicit cash price stated), "medium" (best
    guess from an AED figure not obviously a monthly payment), or
    "none" (no reliable price found - the caller must not invent one).
    """
    if not text:
        return None, "none"

    cash_match = _CASH_PRICE_RE.search(text)
    if cash_match:
        return float(cash_match.group(1).replace(",", "").replace(" ", "")), "high"

    candidates = []
    for match in _GENERIC_AED_RE.finditer(text):
        # Asymmetric window: monthly-payment indicators ("Monthly", "P/M",
        # "cash") always sit immediately after the number in this dataset,
        # so a short forward window is enough - and necessary, because a
        # wide one bleeds into a SECOND, unrelated AED mention nearby
        # (e.g. "AED 89,900 or AED 1,380/Month" would otherwise wrongly
        # exclude the correct 89,900 due to "Month" near the *other*
        # number). Backward stays wide since phrases like "Salary
        # requirement:" precede the number by more than a few characters.
        window = text[max(0, match.start() - 20): match.end() + 12]
        if _MONTHLY_CONTEXT_RE.search(window):
            continue
        candidates.append(float(match.group(1).replace(",", "").replace(" ", "")))

    if candidates:
        return max(candidates), "medium"

    return None, "none"


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="raw dataset")

    # Drop the padding blank rows found in Stage 1 (only real rows have a make)
    df = df.dropna(subset=["make"]).reset_index(drop=True)

    df["listing_id"] = df.index + 1
    df["description_clean"] = df["description"].apply(clean_html)

    price_info = df["description"].fillna("").apply(extract_price)
    df["price_extracted"] = price_info.apply(lambda t: t[0])
    df["price_confidence"] = price_info.apply(lambda t: t[1])

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    columns = [
        "listing_id", "make", "model", "trim", "year", "title",
        "description_clean", "photo_url", "price_extracted", "price_confidence",
    ]
    return df[columns]


if __name__ == "__main__":
    result = load_dataset()
    print(f"Loaded {len(result)} listings")
    print(result["price_confidence"].value_counts())
    print()
    print(result[["make", "model", "year", "price_extracted", "price_confidence"]].head(10))