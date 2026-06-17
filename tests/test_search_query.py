"""Tests for semantic search (mocked LanceDB / fastembed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import Settings
from search.models import PatentRecord
from search.query import embed_query, ensure_index, search_patents


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        release_url="",
        lance_path=tmp_path / "patents.lance",
        top_k=3,
        patentsview_api_key=None,
        checkpoint_path=tmp_path / "checkpoint.json",
        embed_model="BAAI/bge-small-en-v1.5",
        fastembed_model="BAAI/bge-small-en-v1.5",
        max_embed_tokens=512,
        patentsview_api_url="https://example.test/api/v1/patent/",
        cpc_filter="G06N",
        per_page=100,
    )


def test_ensure_index_raises_when_missing(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="LanceDB index not found"):
        ensure_index(settings)


def test_ensure_index_uses_existing_path(settings: Settings) -> None:
    settings.lance_path.mkdir(parents=True, exist_ok=True)
    assert ensure_index(settings) == settings.lance_path


@patch("search.query._get_fastembed_model")
def test_embed_query_returns_vector(mock_model_fn: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed.return_value = [MagicMock(tolist=lambda: [0.1, 0.2, 0.3])]
    mock_model_fn.return_value = mock_model

    vector = embed_query("machine learning")
    assert vector == [0.1, 0.2, 0.3]


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