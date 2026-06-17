"""Semantic search over LanceDB index."""

from __future__ import annotations

import functools
import logging
import shutil
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

logger = logging.getLogger("patenthub.search")

TABLE_NAME = "patents"
OFFLINE_INDEX_MARKER = ".patenthub_offline_index"


class IndexDownloadError(Exception):
    """Failed to download the LanceDB index from a release URL."""


class IndexLoadError(Exception):
    """Failed to open the LanceDB table (missing or corrupt index)."""


def _is_lance_table_dir(path: Path) -> bool:
    """True if path looks like a Lance dataset directory."""
    return path.is_dir() and (
        (path / "_versions").is_dir() or (path / "_transactions").is_dir()
    )


def _resolved_table_dir(
    lance_path: Path, default_table: str = TABLE_NAME
) -> tuple[Path, str] | None:
    """Return (table_dir, table_name) when filesystem layout is recognized."""
    lance_path = lance_path.resolve()

    if _is_lance_table_dir(lance_path):
        return lance_path, lance_path.stem

    nested = lance_path / f"{default_table}.lance"
    if _is_lance_table_dir(nested):
        return nested, default_table

    nested_stem = lance_path / f"{lance_path.stem}.lance"
    if _is_lance_table_dir(nested_stem):
        return nested_stem, lance_path.stem

    return None


def lance_structure_ok(lance_path: Path, default_table: str = TABLE_NAME) -> bool:
    """True when ``lance_path`` contains a recognizable Lance table directory."""
    return _resolved_table_dir(lance_path, default_table) is not None


def resolve_lance_db(
    lance_path: Path,
    default_table: str = TABLE_NAME,
    *,
    strict: bool = False,
) -> tuple[Path, str]:
    """
    Resolve (db_directory, table_name) for lancedb.connect().

    Supports:
    - Flat layout from ingest: ``data/patents.lance`` (table dir) → db ``data/``
    - Nested layout: ``data/patents.lance/patents.lance/`` → db ``data/patents.lance/``

    When ``strict=True`` (used by ``load_index``), raises ``IndexLoadError`` if the
    layout cannot be confirmed via ``_versions`` / ``_transactions`` markers.
    """
    resolved = _resolved_table_dir(lance_path, default_table)
    if resolved is not None:
        table_dir, table_name = resolved
        return table_dir.parent, table_name

    if strict:
        raise IndexLoadError(
            f"No valid LanceDB table structure at {lance_path.resolve()}"
        )

    lance_path = lance_path.resolve()
    if lance_path.is_dir() and lance_path.suffix == ".lance":
        return lance_path.parent, lance_path.stem
    return lance_path.parent, default_table


def _offline_index_message(lance_path: Path) -> str:
    """Return a warning when the index was built with offline pseudo-embeddings."""
    marker = lance_path.resolve() / OFFLINE_INDEX_MARKER
    if marker.exists():
        return (
            "Index de test hors ligne — les résultats ne sont pas sémantiques. "
            "Rebuild without --offline for real embeddings."
        )
    return ""


def _invalid_index_message(path: Path, release_url: str) -> str:
    base = (
        f"Structure d'index invalide à {path}. "
        "Exécutez: python scripts/create_test_index.py --force"
    )
    if release_url:
        base += (
            " — ou supprimez ce répertoire pour permettre le téléchargement "
            "depuis PATENTHUB_RELEASE_URL à la première recherche."
        )
    return base


def index_path_status(settings: Settings | None = None) -> tuple[str, str]:
    """
    Non-blocking index probe (filesystem only, no LanceDB connect).

    Returns (status, message) where status is ``ok``, ``missing``, ``invalid``,
    or ``remote`` (download deferred until search).

    A shallow probe only: a directory passing ``_is_lance_table_dir`` may still
    fail at ``lancedb.connect()`` / ``open_table()`` (corrupt metadata, schema
    mismatch). Full validity is confirmed on the first search.
    """
    settings = settings or get_settings()
    path = settings.lance_path

    if path.exists():
        if lance_structure_ok(path):
            offline_msg = _offline_index_message(path)
            return "ok", offline_msg
        return "invalid", _invalid_index_message(path, settings.release_url)

    if settings.release_url:
        return "remote", "Index distant — téléchargement à la première recherche."

    return (
        "missing",
        "Index local absent. Exécutez: python scripts/create_test_index.py",
    )


def _is_safe_member_path(dest: Path, member_name: str) -> Path:
    """Resolve archive member path and ensure it stays under dest."""
    dest = dest.resolve()
    member_path = (dest / member_name).resolve()
    if not str(member_path).startswith(str(dest)):
        raise IndexDownloadError(f"Unsafe path in archive: {member_name}")
    return member_path


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract tar archive with path-traversal protection."""
    dest = dest.resolve()
    for member in tf.getmembers():
        _is_safe_member_path(dest, member.name)
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest, filter="data")
    else:
        tf.extractall(dest)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip archive with path-traversal protection."""
    dest = dest.resolve()
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        _is_safe_member_path(dest, member)
    zf.extractall(dest)


def _download_release(url: str, dest: Path) -> None:
    """Download and extract index archive from GitHub Release URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(url)
            response.raise_for_status()
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix in {".zip"}:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)
                with zipfile.ZipFile(tmp_path) as zf:
                    _safe_extract_zip(zf, dest.parent)
                tmp_path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)
                with tarfile.open(tmp_path, "r:gz") as tf:
                    _safe_extract_tar(tf, dest.parent)
                tmp_path.unlink(missing_ok=True)
    except httpx.HTTPError as exc:
        raise IndexDownloadError("Impossible de charger l'index") from exc


def ensure_index(settings: Settings | None = None) -> Path:
    """Ensure LanceDB index exists locally; download from release if configured."""
    settings = settings or get_settings()
    lance_path = settings.lance_path

    if lance_path.exists():
        if lance_structure_ok(lance_path):
            return lance_path
        if settings.release_url:
            logger.warning(
                "Removing invalid local index at %s to download from release",
                lance_path,
            )
            shutil.rmtree(lance_path)
        else:
            raise IndexLoadError(_invalid_index_message(lance_path, ""))

    if settings.release_url:
        logger.info("Downloading index from %s", settings.release_url)
        try:
            _download_release(settings.release_url, lance_path)
        except IndexDownloadError:
            raise
        except httpx.HTTPError as exc:
            raise IndexDownloadError("Impossible de charger l'index") from exc
        if lance_path.exists():
            return lance_path
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
    db_dir, table_name = resolve_lance_db(path, strict=True)
    logger.info("Opening LanceDB table=%s db_dir=%s (from %s)", table_name, db_dir, path)
    try:
        db = lancedb.connect(str(db_dir))
        return db.open_table(table_name)
    except Exception as exc:
        raise IndexLoadError(
            f"Cannot open LanceDB table '{table_name}' in {db_dir}: {exc}"
        ) from exc


@functools.lru_cache(maxsize=4)
def get_fastembed_model(model_name: str) -> Any:
    """Load fastembed ONNX model, cached per model name."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


# Backwards-compatible alias for internal callers and tests.
_get_fastembed_model = get_fastembed_model


def embed_query(
    text: str,
    settings: Settings | None = None,
    *,
    model: Any | None = None,
) -> list[float]:
    """Embed a search query using fastembed ONNX (~80MB), cached per model name."""
    settings = settings or get_settings()
    embedder = model or get_fastembed_model(settings.fastembed_model)
    embeddings = list(embedder.embed([text]))
    return embeddings[0].tolist()


def _row_to_record(row: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    record = PatentRecord(
        patent_id=str(row.get("patent_id") or ""),
        title=str(row.get("title") or ""),
        abstract=str(row.get("abstract") or ""),
        claim_snippet=str(row.get("claim_snippet") or ""),
        cpc_codes=list(row.get("cpc_codes") or []),
        filing_date=row.get("filing_date") or None,
        assignee=row.get("assignee") or None,
    )
    result = record.model_dump(mode="json")
    if score is not None:
        result["score"] = score
    return result


def search_patents(
    query: str,
    *,
    top_k: int | None = None,
    table: Any | None = None,
    settings: Settings | None = None,
    embed_model: Any | None = None,
) -> list[dict[str, Any]]:
    """Run vector search and return top-k PatentRecord dicts with scores."""
    settings = settings or get_settings()
    top_k = top_k or settings.top_k
    table = table or load_index(settings)

    vector = embed_query(query, settings, model=embed_model)
    results = table.search(vector).limit(top_k).to_pandas()

    output: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        score = row.get("_distance")
        if score is not None:
            score = float(score)
        output.append(_row_to_record(row.to_dict(), score=score))
    return output