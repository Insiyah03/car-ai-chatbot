# """
# search_inventory() - the 'Retrieval tool'. Hybrid RAG: a structured
# pandas filter (make/model/year/price_extracted) narrows the candidate
# set first, then semantic_search() (Chroma) ranks/handles the fuzzy
# part of the query ("keywords") within that narrowed pool.

# Deliberately importable and testable without any LLM/API dependency:
# the LLM only ever supplies the *arguments* to this function.
# """
# from __future__ import annotations

# from typing import Any, Optional

# import pandas as pd

# from app.data.vector_store import semantic_search

# SEARCH_INVENTORY_SCHEMA = {
#     "type": "function",
#     "function": {
#         "name": "search_inventory",
#         "description": (
#             "Search the dealership's used-car inventory. Combine structured "
#             "filters (make/model/year/price) with free-text 'keywords' for "
#             "fuzzy needs like 'family-friendly' or 'good for road trips'."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "make": {"type": "string", "description": "e.g. 'Toyota', 'Mercedes-Benz'"},
#                 "model": {"type": "string", "description": "e.g. 'Yaris', 'C-Class'"},
#                 "year_min": {"type": "integer", "description": "Earliest model year"},
#                 "year_max": {"type": "integer", "description": "Latest model year"},
#                 "price_max": {"type": "number", "description": "Maximum price in AED"},
#                 "keywords": {
#                     "type": "string",
#                     "description": "Free-text description of desired vibe/use-case for semantic matching",
#                 },
#             },
#             "required": [],
#         },
#     },
# }


# def search_inventory(
#     df: pd.DataFrame,
#     collection: Any,
#     make: Optional[str] = None,
#     model: Optional[str] = None,
#     year_min: Optional[int] = None,
#     year_max: Optional[int] = None,
#     price_max: Optional[float] = None,
#     keywords: Optional[str] = None,
#     limit: int = 5,
# ) -> dict:
#     filtered = df

#     if make:
#         filtered = filtered[filtered["make"].str.lower() == make.lower()]
#     if model:
#         filtered = filtered[filtered["model"].str.lower().str.contains(model.lower(), na=False)]
#     if year_min is not None:
#         filtered = filtered[filtered["year"] >= year_min]
#     if year_max is not None:
#         filtered = filtered[filtered["year"] <= year_max]

#     note = None
#     if price_max is not None:
#         has_price = filtered["price_extracted"].notna()
#         unknown_count = int((~has_price).sum())
#         filtered = filtered[has_price & (filtered["price_extracted"] <= price_max)]
#         if unknown_count:
#             note = (
#                 f"{unknown_count} other listing(s) matched your other filters but had "
#                 "no confirmed price, so they were excluded from this price-filtered list."
#             )

#     candidate_ids = filtered["listing_id"].tolist()

#     if keywords and collection is not None and candidate_ids:
#         ranked_ids = semantic_search(collection, keywords, candidate_ids=candidate_ids, n_results=limit)
#         filtered = filtered.set_index("listing_id").loc[ranked_ids].reset_index()
#     else:
#         filtered = filtered.head(limit)

#     results = filtered.replace({pd.NA: None}).to_dict("records")
#     return {
#         "results": results,
#         "total_matched": len(candidate_ids),
#         "returned": len(results),
#         "note": note,
#     }

"""
search_inventory() - the 'Retrieval tool'. Hybrid RAG: a structured
pandas filter (make/model/year/price_extracted) narrows the candidate
set first, then semantic_search() (Chroma) ranks/handles the fuzzy
part of the query ("keywords") within that narrowed pool.

Deliberately importable and testable without any LLM/API dependency:
the LLM only ever supplies the *arguments* to this function.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from app.data.vector_store import semantic_search

SEARCH_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_inventory",
        "description": (
            "Search the dealership's used-car inventory. Combine structured "
            "filters (make/model/year/price) with free-text 'keywords' for "
            "fuzzy needs like 'family-friendly' or 'good for road trips'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "make": {"type": "string", "description": "e.g. 'Toyota', 'Mercedes-Benz'"},
                "model": {"type": "string", "description": "e.g. 'Yaris', 'C-Class'"},
                "year_min": {"type": "integer", "description": "Earliest model year"},
                "year_max": {"type": "integer", "description": "Latest model year"},
                "price_max": {"type": "number", "description": "Maximum price in AED"},
                "keywords": {
                    "type": "string",
                    "description": "Free-text description of desired vibe/use-case for semantic matching",
                },
            },
            "required": [],
        },
    },
}


def search_inventory(
    df: pd.DataFrame,
    collection: Any,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_max: Optional[float] = None,
    keywords: Optional[str] = None,
    limit: int = 5,
) -> dict:
    filtered = df

    if make:
        filtered = filtered[filtered["make"].str.lower() == make.lower()]
    if model:
        filtered = filtered[filtered["model"].str.lower().str.contains(model.lower(), na=False)]
    if year_min is not None:
        filtered = filtered[filtered["year"] >= year_min]
    if year_max is not None:
        filtered = filtered[filtered["year"] <= year_max]

    note = None
    if price_max is not None:
        has_price = filtered["price_extracted"].notna()
        unknown_count = int((~has_price).sum())
        filtered = filtered[has_price & (filtered["price_extracted"] <= price_max)]
        if unknown_count:
            note = (
                f"{unknown_count} other listing(s) matched your other filters but had "
                "no confirmed price, so they were excluded from this price-filtered list."
            )

    candidate_ids = filtered["listing_id"].tolist()

    if keywords and collection is not None and candidate_ids:
        ranked_ids = semantic_search(collection, keywords, candidate_ids=candidate_ids, n_results=limit)
        filtered = filtered.set_index("listing_id").loc[ranked_ids].reset_index()
    else:
        filtered = filtered.head(limit)

    results = [_json_safe_record(r) for r in filtered.to_dict("records")]
    return {
        "results": results,
        "total_matched": len(candidate_ids),
        "returned": len(results),
        "note": note,
    }


def _json_safe_record(record: dict) -> dict:
    """
    pandas represents a missing regular float (like price_extracted for
    a listing with no confirmed price) as a real float('nan'), and a
    missing nullable-Int64 value (like year) as pd.NA - neither is
    valid JSON, and httpx's json encoder (used by litellm/requests to
    Gemini) correctly refuses to serialize them. Convert both to None
    here, once, rather than letting a raw nan/pd.NA leak into any tool
    result and crash the *next* API call downstream.
    """
    safe = {}
    for key, value in record.items():
        if value is pd.NA:
            safe[key] = None
        elif isinstance(value, float) and math.isnan(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe