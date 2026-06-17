"""Tests for PatentsView fetch, parse, and checkpoint."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from config import Settings
from ingest.fetch import (
    Checkpoint,
    CheckpointError,
    _request_with_backoff,
    fetch_page,
    fetch_patents,
    parse_patent,
    try_parse_patent,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        release_url="",
        lance_path=tmp_path / "patents.lance",
        top_k=10,
        patentsview_api_key=None,
        checkpoint_path=tmp_path / "checkpoint.json",
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        fastembed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
    assert record.filing_date == date(2020, 1, 15)
    assert record.assignee == "Acme AI Corp"


def test_parse_patent_empty_assignee_is_none() -> None:
    raw = {**SAMPLE_PATENT, "assignees": []}
    record = parse_patent(raw)
    assert record.assignee is None


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "cp.json"
    cp = Checkpoint(
        page=3,
        last_page=2,
        fetched_count=2,
        complete=False,
        fetched_ids=["a", "b"],
        records=[{"patent_id": "a"}],
    )
    cp.save(path)
    loaded = Checkpoint.load(path)
    assert loaded.page == 3
    assert loaded.last_page == 2
    assert loaded.fetched_count == 2
    assert loaded.fetched_ids == ["a", "b"]
    assert loaded.complete is False


def test_checkpoint_load_missing_returns_default(tmp_path: Path) -> None:
    loaded = Checkpoint.load(tmp_path / "missing.json")
    assert loaded.page == 1
    assert loaded.fetched_count == 0
    assert loaded.fetched_ids == []


def test_checkpoint_load_corrupt_json_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="Corrupt checkpoint"):
        Checkpoint.load(path)


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_request_with_backoff_retries_on_transient_status(status_code: int) -> None:
    client = MagicMock()
    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"patents": []}
    ok_response.raise_for_status = MagicMock()

    fail_response = MagicMock()
    fail_response.status_code = status_code
    fail_response.request = MagicMock()

    client.post.side_effect = [
        httpx.HTTPStatusError("retry", request=MagicMock(), response=fail_response),
        ok_response,
    ]

    with patch("ingest.fetch.time.sleep"):
        data = _request_with_backoff(
            client, "https://example.test", {"q": {}}, {}, max_retries=3
        )
    assert data == {"patents": []}
    assert client.post.call_count == 2


def test_request_with_backoff_does_not_retry_401() -> None:
    client = MagicMock()
    fail_response = MagicMock()
    fail_response.status_code = 401
    fail_response.request = MagicMock()
    fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=fail_response
    )
    client.post.return_value = fail_response

    with pytest.raises(httpx.HTTPStatusError):
        with patch("ingest.fetch.time.sleep"):
            _request_with_backoff(
                client, "https://example.test", {"q": {}}, {}, max_retries=5
            )
    assert client.post.call_count == 1


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
    payload = client.post.call_args.kwargs["json"]
    assert payload["o"]["page"] == 2
    assert payload["o"]["per_page"] == 2


def test_fetch_patents_paginates(settings: Settings) -> None:
    page_responses = {
        1: {
            "patents": [SAMPLE_PATENT, {**SAMPLE_PATENT, "patent_id": "999"}],
            "total_patent_count": 3,
        },
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
    assert len(cp.fetched_ids) == 3
    assert len(cp.records) == 3


def test_fetch_patents_skips_missing_abstract(settings: Settings) -> None:
    no_abstract = {**SAMPLE_PATENT, "patent_id": "noabs", "patent_abstract": ""}
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "patents": [no_abstract, SAMPLE_PATENT],
        "total_patent_count": 1,
    }
    response.raise_for_status = MagicMock()
    client.post.return_value = response

    records = list(fetch_patents(settings, client=client))
    assert len(records) == 1
    assert records[0].patent_id == "12345678"


def test_try_parse_patent_skips_invalid_record() -> None:
    from pydantic import ValidationError

    with patch("ingest.fetch.parse_patent") as mock_parse:
        mock_parse.side_effect = ValidationError.from_exception_data(
            title="PatentRecord",
            line_errors=[
                {
                    "type": "string_type",
                    "loc": ("patent_id",),
                    "msg": "Input should be a valid string",
                    "input": 12345,
                }
            ],
        )
        assert try_parse_patent(SAMPLE_PATENT) is None