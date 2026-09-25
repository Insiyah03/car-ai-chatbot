"""
Single place all configuration is loaded from. Nothing else in the
codebase should call os.getenv() directly - import `settings` instead.

Deliberately does NOT validate the Gemini API key at import time -
that would mean any module merely importing `settings` (including
tests that never touch the LLM) requires a real key just to load.
Instead, `settings.require_api_key()` is called right before an
actual LLM call is made (in app/llm.py), so a missing key fails at
the point of use, not at import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    data_path: Path = BASE_DIR / "data" / "cars_dataset.xlsx"
    db_path: Path = BASE_DIR / "data" / "app.db"
    leads_csv_path: Path = BASE_DIR / "data" / "leads.csv"
    chroma_dir: Path = BASE_DIR / "data" / ".chroma"

    def require_api_key(self) -> str:
        if not self.gemini_api_key or self.gemini_api_key == "your_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "a real key from https://aistudio.google.com/apikey"
            )
        return self.gemini_api_key


settings = Settings()
