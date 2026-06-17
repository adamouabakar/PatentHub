"""PatentHub semantic search module."""

from search.models import PatentRecord

__all__ = ["PatentRecord", "load_index", "search_patents"]


def __getattr__(name: str):
    """Lazy import query helpers (requires lancedb/fastembed at runtime)."""
    if name in {"load_index", "search_patents"}:
        from search.query import load_index, search_patents

        return load_index if name == "load_index" else search_patents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")