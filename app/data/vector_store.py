"""
Embedded Chroma vector store over listing text, using Gemini
embeddings computed via app.llm.embed_texts (kept outside Chroma's
own embedding_function mechanism so every embedding call goes
through the one retry-aware wrapper, same as chat completions).

This is the RAG half of the hybrid search decided in Stage 4/6:
the structured pandas filter in app/tools/search.py narrows the
candidate set first, then semantic_search() here ranks/handles the
fuzzy part of the query within that narrowed set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib

COLLECTION_NAME = "car_listings"

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get_collection(chroma_dir: Path):
    # Imported lazily: callers that only ever do structured filtering
    # (collection=None passed through search_inventory) never need
    # chromadb importable at all, so a broken/missing Chroma install
    # can't take down search functionality it doesn't actually use.
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME)


# def index_listings(collection, listings: list[dict]) -> None:
#     """
#     listings: dicts with at least listing_id (int) and text (str - the
#     blob to embed, e.g. title + cleaned description). Re-indexes from
#     scratch each run; at ~100 rows this is cheap and avoids stale-
#     embedding bugs from a partially-updated dataset.
#     """
#     if not listings:
#         return
#     from app.llm import embed_texts

#     ids = [str(item["listing_id"]) for item in listings]
#     texts = [item["text"] for item in listings]
#     metadatas = [{"listing_id": item["listing_id"]} for item in listings]
#     embeddings = embed_texts(texts)
#     collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def index_listings(collection, listings: list[dict]) -> None:
    if not listings:
        return
    from app.llm import embed_texts

    ids = [str(item["listing_id"]) for item in listings]
    texts = [item["text"] for item in listings]
    hashes = [_content_hash(t) for t in texts]

    # Check what's already indexed with matching content
    existing = collection.get(ids=ids, include=["metadatas"])
    existing_hashes = {
        eid: meta.get("content_hash")
        for eid, meta in zip(existing["ids"], existing["metadatas"])
    }

    to_index = [
        i for i, (item_id, h) in enumerate(zip(ids, hashes))
        if existing_hashes.get(item_id) != h
    ]

    if not to_index:
        return  # nothing changed, skip embedding entirely

    ids_sub = [ids[i] for i in to_index]
    texts_sub = [texts[i] for i in to_index]
    metadatas_sub = [
        {"listing_id": listings[i]["listing_id"], "content_hash": hashes[i]}
        for i in to_index
    ]
    embeddings = embed_texts(texts_sub)
    collection.upsert(ids=ids_sub, documents=texts_sub, embeddings=embeddings, metadatas=metadatas_sub)

def semantic_search(
    collection,
    query: str,
    candidate_ids: Optional[list[int]] = None,
    n_results: int = 5,
) -> list[int]:
    """
    Returns listing_ids ranked by semantic similarity to `query`.
    If candidate_ids is given (the structured pre-filter's output),
    the search is restricted to just that pool via a metadata filter -
    this is what makes it "hybrid" rather than pure semantic search.
    Returns [] if the candidate pool is empty (nothing to rank).
    """
    if candidate_ids is not None and len(candidate_ids) == 0:
        return []
    from app.llm import embed_texts

    query_embedding = embed_texts([query])[0]
    where = {"listing_id": {"$in": candidate_ids}} if candidate_ids is not None else None
    n = min(n_results, len(candidate_ids)) if candidate_ids is not None else n_results

    results = collection.query(query_embeddings=[query_embedding], n_results=n, where=where)
    return [int(i) for i in results["ids"][0]]
