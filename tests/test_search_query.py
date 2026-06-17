"""Tests for semantic search (mocked LanceDB / fastembed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from config import Settings
from search.query import (
    IndexDownloadError,
    IndexLoadError,
    OFFLINE_INDEX_MARKER,
    _get_fastembed_model,
    _is_lance_table_dir,
    embed_query,
    ensure_index,
    get_fastembed_model,
    index_path_status,
    load_index,
    resolve_lance_db,
    search_patents,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        release_url="https://example.test/release.tar.gz",
        lance_path=tmp_path / "patents.lance",
        top_k=3,
        patentsview_api_key=None,
        checkpoint_path=tmp_path / "checkpoint.json",
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        fastembed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_embed_tokens=512,
        patentsview_api_url="https://example.test/api/v1/patent/",
        cpc_filter="G06N",
        per_page=100,
    )


def test_ensure_index_raises_when_missing(settings: Settings) -> None:
    settings = Settings(
        release_url="",
        lance_path=settings.lance_path,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    with pytest.raises(FileNotFoundError, match="LanceDB index not found"):
        ensure_index(settings)


def test_ensure_index_uses_existing_path(settings: Settings) -> None:
    (settings.lance_path / "_versions").mkdir(parents=True)
    assert ensure_index(settings) == settings.lance_path


def test_ensure_index_raises_on_corrupt_local_path(settings: Settings) -> None:
    settings = Settings(
        release_url="",
        lance_path=settings.lance_path,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    settings.lance_path.mkdir(parents=True)
    with pytest.raises(IndexLoadError, match="Structure d'index invalide"):
        ensure_index(settings)


@patch("search.query._download_release")
def test_ensure_index_removes_corrupt_path_and_downloads(
    mock_download: MagicMock, settings: Settings
) -> None:
    settings.lance_path.mkdir(parents=True)
    mock_download.return_value = None

    def _create_valid_index(_url: str, dest: Path) -> None:
        (dest / "_versions").mkdir(parents=True)

    mock_download.side_effect = _create_valid_index
    path = ensure_index(settings)
    assert path == settings.lance_path
    assert _is_lance_table_dir(settings.lance_path)
    mock_download.assert_called_once()


def test_is_lance_table_dir_detects_versions(tmp_path: Path) -> None:
    table_dir = tmp_path / "patents.lance"
    table_dir.mkdir()
    (table_dir / "_versions").mkdir()
    assert _is_lance_table_dir(table_dir) is True
    assert _is_lance_table_dir(tmp_path / "empty") is False


def test_resolve_lance_db_flat_layout(tmp_path: Path) -> None:
    table_dir = tmp_path / "patents.lance"
    (table_dir / "_versions").mkdir(parents=True)
    db_dir, table_name = resolve_lance_db(table_dir)
    assert db_dir == tmp_path
    assert table_name == "patents"


def test_resolve_lance_db_nested_layout(tmp_path: Path) -> None:
    root = tmp_path / "patents.lance"
    nested = root / "patents.lance"
    (nested / "_versions").mkdir(parents=True)
    db_dir, table_name = resolve_lance_db(root)
    assert db_dir == root
    assert table_name == "patents"


def test_resolve_lance_db_custom_stem_nested(tmp_path: Path) -> None:
    root = tmp_path / "foo.lance"
    nested = root / "foo.lance"
    (nested / "_versions").mkdir(parents=True)
    db_dir, table_name = resolve_lance_db(root)
    assert db_dir == root
    assert table_name == "foo"


def test_resolve_lance_db_strict_raises_on_unknown_layout(tmp_path: Path) -> None:
    bad = tmp_path / "patents.lance"
    bad.mkdir()
    with pytest.raises(IndexLoadError, match="No valid LanceDB table structure"):
        resolve_lance_db(bad, strict=True)


def test_index_path_status_missing(settings: Settings) -> None:
    settings = Settings(
        release_url="",
        lance_path=settings.lance_path,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "missing"
    assert "create_test_index" in msg


def test_index_path_status_ok_nested(tmp_path: Path, settings: Settings) -> None:
    root = tmp_path / "patents.lance"
    (root / "patents.lance" / "_versions").mkdir(parents=True)
    settings = Settings(
        release_url="",
        lance_path=root,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "ok"
    assert msg == ""


def test_index_path_status_invalid(tmp_path: Path, settings: Settings) -> None:
    corrupt = tmp_path / "patents.lance"
    corrupt.mkdir()
    settings = Settings(
        release_url="",
        lance_path=corrupt,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "invalid"
    assert "create_test_index.py --force" in msg


def test_index_path_status_invalid_with_release_url(tmp_path: Path, settings: Settings) -> None:
    corrupt = tmp_path / "patents.lance"
    corrupt.mkdir()
    settings = Settings(
        release_url="https://example.test/release.tar.gz",
        lance_path=corrupt,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "invalid"
    assert "PATENTHUB_RELEASE_URL" in msg


def test_index_path_status_remote(settings: Settings) -> None:
    status, msg = index_path_status(settings)
    assert status == "remote"
    assert "téléchargement" in msg


def test_index_path_status_offline_marker(tmp_path: Path, settings: Settings) -> None:
    lance_path = tmp_path / "patents.lance"
    (lance_path / "_versions").mkdir(parents=True)
    (lance_path / OFFLINE_INDEX_MARKER).write_text("offline\n", encoding="utf-8")
    settings = Settings(
        release_url="",
        lance_path=lance_path,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "ok"
    assert "hors ligne" in msg


def test_index_path_status_ok_flat(tmp_path: Path, settings: Settings) -> None:
    lance_path = tmp_path / "patents.lance"
    (lance_path / "_versions").mkdir(parents=True)
    settings = Settings(
        release_url="",
        lance_path=lance_path,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )
    status, msg = index_path_status(settings)
    assert status == "ok"
    assert msg == ""


@patch("search.query.lancedb.connect")
def test_load_index_uses_nested_resolution(
    mock_connect: MagicMock, tmp_path: Path, settings: Settings
) -> None:
    root = tmp_path / "patents.lance"
    nested = root / "patents.lance"
    (nested / "_versions").mkdir(parents=True)

    settings = Settings(
        release_url="",
        lance_path=root,
        top_k=settings.top_k,
        patentsview_api_key=None,
        checkpoint_path=settings.checkpoint_path,
        embed_model=settings.embed_model,
        fastembed_model=settings.fastembed_model,
        max_embed_tokens=settings.max_embed_tokens,
        patentsview_api_url=settings.patentsview_api_url,
        cpc_filter=settings.cpc_filter,
        per_page=settings.per_page,
    )

    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_connect.return_value = mock_db
    mock_db.open_table.return_value = mock_table

    table = load_index(settings)
    assert table is mock_table
    mock_connect.assert_called_once_with(str(root))
    mock_db.open_table.assert_called_once_with("patents")


@patch("search.query.ensure_index")
@patch("search.query.lancedb.connect")
def test_load_index_raises_index_load_error(
    mock_connect: MagicMock,
    mock_ensure: MagicMock,
    settings: Settings,
) -> None:
    mock_ensure.return_value = settings.lance_path
    (settings.lance_path / "_versions").mkdir(parents=True)
    mock_db = MagicMock()
    mock_connect.return_value = mock_db
    mock_db.open_table.side_effect = ValueError("Table 'patents' was not found")

    with pytest.raises(IndexLoadError, match="Cannot open LanceDB table"):
        load_index(settings)


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    import io
    import zipfile

    from search.query import IndexDownloadError, _safe_extract_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "bad")
    buf.seek(0)

    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(IndexDownloadError, match="Unsafe path"):
            _safe_extract_zip(zf, tmp_path)


@patch("search.query._download_release")
def test_ensure_index_download_failure_raises_index_download_error(
    mock_download: MagicMock, settings: Settings
) -> None:
    mock_download.side_effect = httpx.HTTPStatusError(
        "404",
        request=MagicMock(),
        response=MagicMock(status_code=404),
    )
    with pytest.raises(IndexDownloadError, match="Impossible de charger l'index"):
        ensure_index(settings)


def test_get_fastembed_model_is_lru_cached() -> None:
    import sys

    import search.query as query_mod

    query_mod.get_fastembed_model.cache_clear()
    mock_model = MagicMock()
    mock_cls = MagicMock(return_value=mock_model)
    fake_fastembed = MagicMock(TextEmbedding=mock_cls)
    with patch.dict(sys.modules, {"fastembed": fake_fastembed}):
        first = query_mod.get_fastembed_model(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        second = query_mod.get_fastembed_model(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    assert first is second
    mock_cls.assert_called_once()


@patch("search.query.get_fastembed_model")
def test_embed_query_returns_vector(mock_model_fn: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed.return_value = [MagicMock(tolist=lambda: [0.1, 0.2, 0.3])]
    mock_model_fn.return_value = mock_model

    vector = embed_query("machine learning")
    assert vector == [0.1, 0.2, 0.3]
    mock_model_fn.assert_called_once()


@patch("search.query.embed_query")
def test_search_patents_returns_records(
    mock_embed: MagicMock, settings: Settings
) -> None:
    mock_embed.return_value = [0.5, 0.5, 0.5]

    df = pd.DataFrame(
        [
            {
                "patent_id": "111",
                "title": "Neural Net",
                "abstract": "AI abstract",
                "claim_snippet": "claim text",
                "cpc_codes": ["G06N"],
                "filing_date": "2019-05-01",
                "assignee": "Test Corp",
                "_distance": 0.12,
            }
        ]
    )

    mock_table = MagicMock()
    mock_search = MagicMock()
    mock_table.search.return_value = mock_search
    mock_search.limit.return_value = mock_search
    mock_search.to_pandas.return_value = df

    results = search_patents("réseaux de neurones", table=mock_table, settings=settings)
    assert len(results) == 1
    assert results[0]["patent_id"] == "111"
    assert results[0]["title"] == "Neural Net"
    assert results[0]["score"] == pytest.approx(0.12)
    mock_table.search.assert_called_once_with([0.5, 0.5, 0.5])
    mock_search.limit.assert_called_once_with(3)