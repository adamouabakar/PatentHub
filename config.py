"""PatentHub configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment."""

    release_url: str
    lance_path: Path
    top_k: int
    patentsview_api_key: str | None
    checkpoint_path: Path
    embed_model: str
    fastembed_model: str
    max_embed_tokens: int
    patentsview_api_url: str
    cpc_filter: str
    per_page: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            release_url=os.environ.get("PATENTHUB_RELEASE_URL", ""),
            lance_path=Path(
                os.environ.get("PATENTHUB_LANCE_PATH", "./data/patents.lance")
            ),
            top_k=int(os.environ.get("PATENTHUB_TOP_K", "10")),
            patentsview_api_key=os.environ.get("PATENTSVIEW_API_KEY") or None,
            checkpoint_path=Path(
                os.environ.get(
                    "PATENTHUB_CHECKPOINT_PATH", "./data/ingest_checkpoint.json"
                )
            ),
            embed_model=os.environ.get(
                "PATENTHUB_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            fastembed_model=os.environ.get(
                "PATENTHUB_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            max_embed_tokens=int(os.environ.get("PATENTHUB_MAX_EMBED_TOKENS", "512")),
            patentsview_api_url=os.environ.get(
                "PATENTSVIEW_API_URL",
                "https://search.patentsview.org/api/v1/patent/",
            ),
            cpc_filter=os.environ.get("PATENTHUB_CPC_FILTER", "G06N"),
            per_page=int(os.environ.get("PATENTHUB_PER_PAGE", "100")),
        )


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings.from_env()