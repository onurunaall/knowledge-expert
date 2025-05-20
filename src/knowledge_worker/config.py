"""
Centralised, read-only configuration for the Knowledge Expert project.

• Resolves project paths relative to this file.
• Pulls required settings from environment variables (via .env).
• Exposes a single validated `settings` object plus key path constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
# <project-root>/src/knowledge_worker/config.py  → two parents up is <project-root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_DIR: Path = PROJECT_ROOT / ".vector_store"

# Ensure directories exist at import-time (idempotent, avoids runtime checks)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Environment variables                                                       #
# --------------------------------------------------------------------------- #

load_dotenv()  # read .env if present (ignored if missing)


@dataclass(frozen=True, slots=True)
class _Settings:
    """Immutable, validated runtime settings."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    def __post_init__(self) -> None:  # noqa: D401
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )


# Public, validated settings instance
settings = _Settings()

__all__ = [
    "PROJECT_ROOT",
    "KNOWLEDGE_BASE_DIR",
    "VECTOR_STORE_DIR",
    "settings",
]
