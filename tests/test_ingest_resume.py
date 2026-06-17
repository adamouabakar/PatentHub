"""
CRITICAL: test checkpoint resume after simulated API failure mid-ingest.

Simulates page 1 success, page 2 failure, then resume from checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from config import Settings
from ingest.build_index import CheckpointCompleteError, build_index
from ingest.fetch import Checkpoint, fetch_patents

SAMPLE = {
    "patent_id": "10000001",
    "patent_title": "ML Patent",
    "patent_abstract": "Abstract",
    "patent_date": "2021-01-01",
    "assignees": [],
    "cpc_current": [{"cpc_subgroup_id": "G06N"}],
    "claims": [{"claim_text": "1. A method."}],
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        release_url="",
        lance_path=tmp_path / "patents.lance",
        top_k=10,
        patentsview_api_key=None,
        checkpoint_path=tmp_path / "ingest_checkpoint.json",
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        fastembed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_embed_tokens=512,
        patentsview_api_url="https://example.test/api/v1/patent/",
        cpc_filter="G06N",
        per_page=2,
    )


def _make_client(pages: dict[int, dict], fail_on_page: int | None = None) -> MagicMock:
    """Return mock httpx client; optionally raise on a specific page."""
    client = MagicMock()

    def mock_post(url, json=None, headers=None, timeout=None):
        page = json["o"]["page"]
        if fail_on_page is not None and page == fail_on_page:
            raise httpx.TransportError("Simulated network failure")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = pages.get(
            page, {"patents": [], "total_patent_count": 4}
        )
        resp.raise_for_status = MagicMock()
        return resp

    client.post.side_effect = mock_post
    return client


def test_ingest_resume_after_simulated_failure(settings: Settings) -> None:
    """
    Page 1: 2 patents saved, checkpoint page=2.
    Page 2: simulated failure — checkpoint must persist page=2.
    Resume: continues from page 2 without re-fetching page 1 duplicates.
    """
    pages = {
        1: {
            "patents": [
                {**SAMPLE, "patent_id": "10000001"},
                {**SAMPLE, "patent_id": "10000002"},
            ],
            "total_patent_count": 4,
        },
        2: {
            "patents": [
                {**SAMPLE, "patent_id": "10000003"},
                {**SAMPLE, "patent_id": "10000004"},
            ],
            "total_patent_count": 4,
        },
        3: {"patents": [], "total_patent_count": 4},
    }

    client = _make_client(pages, fail_on_page=2)

    with pytest.raises(httpx.TransportError):
        list(fetch_patents(settings, client=client))

    cp = Checkpoint.load(settings.checkpoint_path)
    assert cp.page == 2, "checkpoint must point to failed page"
    assert cp.fetched_count == 2, "page 1 records counted"
    assert cp.fetched_ids == ["10000001", "10000002"]
    assert cp.complete is False

    client2 = _make_client(pages)
    records = list(fetch_patents(settings, client=client2))

    assert len(records) == 2, "resume yields only remaining patents"
    assert {r.patent_id for r in records} == {"10000003", "10000004"}

    requested_pages = [
        call.kwargs["json"]["o"]["page"] for call in client2.post.call_args_list
    ]
    assert 1 not in requested_pages

    cp_final = Checkpoint.load(settings.checkpoint_path)
    assert cp_final.complete is True
    assert cp_final.fetched_count == 4


def test_ingest_resume_with_limit(settings: Settings) -> None:
    """Limit stops ingest and marks checkpoint complete."""
    pages = {
        1: {
            "patents": [
                {**SAMPLE, "patent_id": "A"},
                {**SAMPLE, "patent_id": "B"},
            ],
            "total_patent_count": 100,
        },
    }
    client = _make_client(pages)
    records = list(fetch_patents(settings, limit=1, client=client))
    assert len(records) == 1
    cp = Checkpoint.load(settings.checkpoint_path)
    assert cp.complete is True
    assert cp.fetched_count == 1


def test_mid_page_resume_skips_duplicate_ids(settings: Settings) -> None:
    """If one record from a page was checkpointed before failure, resume skips it."""
    from ingest.fetch import parse_patent

    pages = {
        1: {
            "patents": [
                {**SAMPLE, "patent_id": "dup1"},
                {**SAMPLE, "patent_id": "dup2"},
            ],
            "total_patent_count": 2,
        },
    }
    partial = parse_patent(pages[1]["patents"][0])
    cp = Checkpoint(
        page=1,
        fetched_ids=["dup1"],
        fetched_count=1,
        records=[partial.model_dump(mode="json")],
    )
    cp.save(settings.checkpoint_path)

    client = _make_client(pages)
    records = list(fetch_patents(settings, client=client))
    assert [r.patent_id for r in records] == ["dup2"]


@patch("ingest.build_index._append_batch")
@patch("ingest.build_index.embed_texts_offline")
@patch("ingest.build_index._embedded_ids")
def test_build_index_resume_after_fetch_failure(
    mock_embedded_ids: MagicMock,
    mock_embed: MagicMock,
    mock_append: MagicMock,
    settings: Settings,
) -> None:
    """build_index retains checkpointed page-1 records, then appends page-2 on resume."""
    stored_ids: set[str] = set()

    mock_embed.side_effect = lambda texts, model, batch_size=32: [
        [float(i), 0.1, 0.2] for i in range(len(texts))
    ]
    mock_embedded_ids.side_effect = lambda db, table_name, table_dir: set(stored_ids)

    def track_append(
        db, table_name, table_dir, records, texts, vectors
    ) -> None:
        stored_ids.update(r.patent_id for r in records)

    mock_append.side_effect = track_append

    pages = {
        1: {
            "patents": [
                {**SAMPLE, "patent_id": "10000001"},
                {**SAMPLE, "patent_id": "10000002"},
            ],
            "total_patent_count": 4,
        },
        2: {
            "patents": [
                {**SAMPLE, "patent_id": "10000003"},
                {**SAMPLE, "patent_id": "10000004"},
            ],
            "total_patent_count": 4,
        },
        3: {"patents": [], "total_patent_count": 4},
    }

    client = _make_client(pages, fail_on_page=2)
    with pytest.raises(httpx.TransportError):
        list(fetch_patents(settings, client=client))

    with patch("ingest.build_index.fetch_patents", return_value=iter([])):
        build_index(settings=settings, output_path=settings.lance_path, batch_size=2)

    assert stored_ids == {"10000001", "10000002"}

    client2 = _make_client(pages)
    with patch(
        "ingest.build_index.fetch_patents",
        side_effect=lambda *a, **k: fetch_patents(settings, client=client2),
    ):
        build_index(settings=settings, output_path=settings.lance_path, batch_size=2)

    assert stored_ids == {"10000001", "10000002", "10000003", "10000004"}


def test_build_index_raises_when_checkpoint_complete(settings: Settings) -> None:
    cp = Checkpoint(complete=True, fetched_ids=["x"], records=[{"patent_id": "x"}])
    cp.save(settings.checkpoint_path)

    with pytest.raises(CheckpointCompleteError, match="--reset-checkpoint"):
        build_index(settings=settings, output_path=settings.lance_path)