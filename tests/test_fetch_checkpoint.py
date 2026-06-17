"""Tests for PatentsView fetch, parse, and checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from config import Settings
from ingest.fetch import (
    Checkpoint,
    _request_with_backoff,
    fetch_page,
    fetch_patents,
    parse_patent,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        release_url="",
        lance_path=tmp_path / "patents.lance",
        top_k=10,
        patentsview_api_key=None,
        checkpoint_path=tmp_path / "checkpoint.json",
        embed_model="BAAI/bge-small-en-v1.5",
        fastembed_model="BAAI/bge-small-en-v1.5",
        max_embed_tokens=512,
        patentsview_api_url="https://example.test/api/v1/patent/",
        cpc_filter="G06N",
        per_page=2,
    )


SAMPLE_PATENT = {
    "patent_id": "12345678",
    "patent_title": "Neural network system",
    "patent_abstract": "A method for training models.",
    "patent_date": "2020-01-15",
    "assignees": [{"assignee_organization": "Acme AI Corp"}],
    "cpc_current": [{"cpc_subgroup_id": "G06N3/08"}],
    "claims": [{"claim_text": "1. A computer-implemented method comprising..."}],
}


def test_parse_patent_maps_fields() -> None:
    record = parse_patent(SAMPLE_PATENT)
    assert record.patent_id == "12345678"
    assert record.title == "Neural network system"
    assert "training" in record.abstract
    assert "computer-implemented" in record.claim_snippet
    assert record.cpc_codes == ["G06N3/08"]
    assert record.filing_date == "2020-01-15"
    assert record.assignee == "Acme AI Corp"


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "cp.json"
    cp = Checkpoint(page=3, fetched_count=150, complete=False)
    cp.save(path)
    loaded = Checkpoint.load(path)
    assert loaded.page == 3
    assert loaded.fetched_count == 150
    assert loaded.complete is False


def test_checkpoint_load_missing_returns_default(tmp_path: Path) -> None:
    loaded = Checkpoint.load(tmp_path / "missing.json")
    assert loaded.page == 1
    assert loaded.fetched_count == 0


def test_request_with_backoff_retries_on_429() -> None:
    client = MagicMock()
    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"patents": []}
    ok_response.raise_for_status = MagicMock()

    fail_response = MagicMock()
    fail_response.status_code = 429
    fail_response.request = MagicMock()

    client.post.side_effect = [
        httpx.HTTPStatusError("429", request=MagicMock(), response=fail_response),
        ok_response,
    ]

    with patch("ingest.fetch.time.sleep"):
        data = _request_with_backoff(
            client, "https://example.test", {"q": {}}, {}, max_retries=3
        )
    assert data == {"patents": []}
    assert client.post.call_count == 2


def test_fetch_page_builds_payload(settings: Settings) -> None:
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "patents": [SAMPLE_PATENT],
        "total_patent_count": 1,
    }
    response.raise_for_status = MagicMock()
    client.post.return_value = response

    patents, total = fetch_page(client, settings, page=2)
    assert len(patents) == 1
    assert total == 1
    call_kwargs = client.post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["o"]["page"] == 2
    assert payload["o"]["per_page"] == 2


def test_fetch_patents_paginates(settings: Settings) -> None:
    page_responses = {
        1: {"patents": [SAMPLE_PATENT, {**SAMPLE_PATENT, "patent_id": "999"}], "total_patent_count": 3},
        2: {"patents": [{**SAMPLE_PATENT, "patent_id": "888"}], "total_patent_count": 3},
        3: {"patents": [], "total_patent_count": 3},
    }

    def mock_post(url, json=None, headers=None, timeout=None):
        page = json["o"]["page"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = page_responses[page]
        resp.raise_for_status = MagicMock()
        return resp

    client = MagicMock()
    client.post.side_effect = mock_post

    records = list(fetch_patents(settings, client=client))
    assert len(records) == 3
    assert {r.patent_id for r in records} == {"12345678", "999", "888"}

    cp = Checkpoint.load(settings.checkpoint_path)
    assert cp.complete is True
    assert cp.fetched_count == 3