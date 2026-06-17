"""Text preparation and offline embedding for ingest."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from search.models import PatentRecord

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def truncate_to_tokens(text: str, max_tokens: int = 512) -> str:
    """Approximate token truncation (whitespace split, safe for CI without tokenizer)."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def build_embed_text(record: PatentRecord, max_tokens: int = 512) -> str:
    """Build title + abstract + claim[0] text truncated to max_tokens."""
    return truncate_to_tokens(record.embed_text(), max_tokens=max_tokens)


def load_sentence_transformer(model_name: str) -> "SentenceTransformer":
    """Lazy-load sentence-transformers model for offline ingest."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts_offline(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed texts using sentence-transformers (offline ingest)."""
    model = load_sentence_transformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return [vec.tolist() for vec in vectors]