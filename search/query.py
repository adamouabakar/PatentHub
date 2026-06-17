"""Semantic search over LanceDB index."""

from __future__ import annotations

import logging
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import lancedb

from config import Settings, get_settings
from search.models import PatentRecord

logger = logging.getLogger(__name__)

TABLE_NAME = "patents"


def _download_release(url: str, dest: Path) -> None:
    """Download and extract index archive from GitHub Release URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in {".zip"}:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(dest.parent)
            tmp_path.unlink(missing_ok=True)
        else:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)
            with tarfile.open(tmp_path, "r:gz") as tf:
                tf.extractall(dest.parent)
            tmp_path.unlink(missing_ok=True)


def ensure_index(settings: Settings | None = None) -> Path:
    """Ensure LanceDB index exists locally; download from release if configured."""
    settings = settings or get_settings()
    lance_path = settings.lance_path

    if lance_path.exists():
        return lance_path

    if settings.release_url:
        logger.info("Downloading index from %s", settings.release_url)
        _download_release(settings.release_url, lance_path)
        if lance_path.exists():
            return lance_path
        # Archive may extract to patents.lance under data/
        candidate = lance_path.parent / "patents.lance"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"LanceDB index not found at {lance_path}. "
        "Set PATENTHUB_RELEASE_URL or run ingest/build_index.py locally."
    )


def load_index(settings: Settings | None = None) -> Any:
    """Load LanceDB table (use with @st.cache_resource in Streamlit)."""
    settings = settings or get_settings()
    path = ensure_index(settings)
    if path.is_dir() and path.suffix == ".lance":
        db_dir = path.parent
        table_name = path.stem
    else:
        db_dir = path.parent
        table_name = path.stem if path.suffix == ".lance" else TABLE_NAME
    db = lancedb.connect(str(db_dir))
    return db.open_table(table_name)


def _get_fastembed_model(model_name: str) -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a search query using fastembed ONNX (~80MB)."""
    settings = settings or get_settings()
    model = _get_fastembed_model(settings.fastembed_model)
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def _row_to_record(row: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    record = PatentRecord(
        patent_id=str(row.get("patent_id") or ""),
        title=str(row.get("title") or ""),
        abstract=str(row.get("abstract") or ""),
        claim_snippet=str(row.get("claim_snippet") or ""),
        cpc_codes=list(row.get("cpc_codes") or []),
        filing_date=str(row.get("filing_date") or ""),
        assignee=str(row.get("assignee") or ""),
    )
    result = record.model_dump()
    if score is not None:
        result["score"] = score
    return result


def search_patents(
    query: str,
    *,
    top_k: int | None = None,
    table: Any | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Run vector search and return top-k PatentRecord dicts with scores."""
    settings = settings or get_settings()
    top_k = top_k or settings.top_k
    table = table or load_index(settings)

    vector = embed_query(query, settings)
    results = (
        table.search(vector)
        .limit(top_k)
        .to_pandas()
    )

    output: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        score = row.get("_distance")
        if score is not None:
            score = float(score)
        output.append(_row_to_record(row.to_dict(), score=score))
    return output