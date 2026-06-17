"""
CRITICAL: test checkpoint resume after simulated API failure mid-ingest.

Simulates page 1 success, page 2 failure, then resume from checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from config import Settings
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
        embed_model="BAAI/bge-small-en-v1.5",
        fastembed_model="BAAI/bge-small-en-v1.5",
        max_embed_tokens=512,
        patentsview_api_url="https://example.test/api/v1/patent/",
        cpc_filter="G06N",
        per_page=2,
    )


def _make_client(pages: dict[int, dict], fail_on_page: int | None = None) -> MagicMock:
    """Return mock httpx client; optionally raise on a specific page."""
    client = MagicMock()
    call_count = {"n": 0}

    def mock_post(url, json=None, headers=None, timeout=None):
        page = json["o"]["page"]
        call_count["n"] += 1
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

    # First run: fails on page 2 after yielding page 1 records
    with pytest.raises(httpx.TransportError):
        list(fetch_patents(settings, client=client))

    cp = Checkpoint.load(settings.checkpoint_path)
    assert cp.page == 2, "checkpoint must point to failed page"
    assert cp.fetched_count == 2, "page 1 records counted"
    assert cp.complete is False

    # Resume with fresh client (no failure)
    client2 = _make_client(pages)
    records = list(fetch_patents(settings, client=client2))

    assert len(records) == 2, "resume yields only remaining patents"
    assert {r.patent_id for r in records} == {"10000003", "10000004"}

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