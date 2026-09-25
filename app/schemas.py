"""
Pydantic models = the API contract between the Streamlit client and
FastAPI. Deliberately minimal - just what /chat and /health need.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Client-persisted long-term identity")
    session_id: str = Field(..., min_length=1, description="Per-session identity (short-term memory scope)")
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    status: str
    listings_loaded: int
    hybrid_search_available: bool
