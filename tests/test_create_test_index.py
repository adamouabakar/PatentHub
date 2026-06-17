"""Integration tests for scripts/create_test_index.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from config import Settings
from search.query import OFFLINE_INDEX_MARKER, _is_lance_table_dir, load_index

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.create_test_index import create_test_index  # noqa: E402


def _settings(tmp_path: Path, lance_path: Path) -> Settings:
    return Settings(
        release_url="",
        lance_path=lance_path,
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


def test_create_test_index_offline_force(tmp_path: Path) -> None:
    lance_path = tmp_path / "patents.lance"
    settings = _settings(tmp_path, lance_path)

    table_dir = create_test_index(lance_path, force=True, offline=True, settings=settings)

    assert _is_lance_table_dir(table_dir)
    assert (lance_path / OFFLINE_INDEX_MARKER).exists()
    table = load_index(settings)
    assert table.count_rows() == 5


def test_create_test_index_nested_without_force_raises(tmp_path: Path) -> None:
    root = tmp_path / "patents.lance"
    (root / "patents.lance" / "_versions").mkdir(parents=True)
    settings = _settings(tmp_path, root)

    with pytest.raises(RuntimeError, match="--force"):
        create_test_index(root, force=False, offline=True, settings=settings)


def test_create_test_index_skips_existing_flat(tmp_path: Path) -> None:
    lance_path = tmp_path / "patents.lance"
    settings = _settings(tmp_path, lance_path)

    create_test_index(lance_path, force=True, offline=True, settings=settings)
    table_dir = create_test_index(lance_path, force=False, offline=True, settings=settings)

    assert table_dir == lance_path
    table = load_index(settings)
    assert table.count_rows() == 5


def test_create_test_index_nested_force_replaces(tmp_path: Path) -> None:
    root = tmp_path / "patents.lance"
    (root / "patents.lance" / "_versions").mkdir(parents=True)
    settings = _settings(tmp_path, root)

    table_dir = create_test_index(root, force=True, offline=True, settings=settings)

    assert _is_lance_table_dir(table_dir)
    assert not (root / "patents.lance").exists()  # nested layout removed
    table = load_index(settings)
    assert table.count_rows() == 5