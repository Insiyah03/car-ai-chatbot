"""
FastAPI entrypoint. Deliberately thin: route definitions + one-time
startup wiring only - all real logic lives in app/orchestrator.py
and friends. Run with: uv run uvicorn main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.data.loader import load_dataset
from app.data.vector_store import get_collection, index_listings
from app.orchestrator import handle_message
from app.persistence.db import init_db
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything expensive/shared happens once here, not per-request -
    # the dataset, DB schema, and vector index are all read-mostly and
    # cheap to hold in memory at this scale (~100 rows). Stored on
    # app.state (FastAPI's own idiomatic pattern), not a module-level
    # global, so multiple app instances (e.g. in tests) don't collide.
    logger.info("Loading dataset...")
    df = load_dataset(settings.data_path)
    app.state.df = df

    logger.info("Initializing database...")
    init_db(settings.db_path)

    logger.info("Building vector index for hybrid search...")
    try:
        collection = get_collection(settings.chroma_dir)
        listings = [
            {"listing_id": int(row.listing_id), "text": f"{row.title} {row.description_clean}"}
            for row in df.itertuples()
        ]
        index_listings(collection, listings)
        app.state.collection = collection
        logger.info("Vector index ready (%d listings).", len(listings))
    except Exception:
        # Degrade gracefully to structured-only search rather than
        # refusing to start the whole service over an embedding/quota
        # failure (Stage 5 reliability decision).
        logger.exception("Vector index build failed - falling back to structured-only search.")
        app.state.collection = None

    logger.info("Startup complete: %d listings loaded.", len(df))
    yield
    # Nothing to clean up explicitly - sqlite connections are opened
    # and closed per-call (app/persistence/db.py), not held open here.


app = FastAPI(title="dubizzle Car Assistant", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    df = getattr(request.app.state, "df", None)
    collection = getattr(request.app.state, "collection", None)
    return HealthResponse(
        status="ok",
        listings_loaded=len(df) if df is not None else 0,
        hybrid_search_available=collection is not None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    df = getattr(request.app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Service is still starting up - try again shortly.")

    try:
        reply = handle_message(
            df=df,
            collection=getattr(request.app.state, "collection", None),
            user_id=payload.user_id,
            session_id=payload.session_id,
            user_message=payload.message,
        )
    except Exception:
        # Never leak internal stack traces to the client (Stage 5
        # security decision) - log the real error, return a generic one.
        logger.exception("Unhandled error handling chat message")
        raise HTTPException(status_code=500, detail="Something went wrong processing your message.")

    return ChatResponse(reply=reply)
