"""Fetch G06N patents from PatentsView API with retry and checkpoint."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
from pydantic import ValidationError

from config import Settings, get_settings
from search.models import PatentRecord

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

PATENT_FIELDS = [
    "patent_id",
    "patent_title",
    "patent_abstract",
    "patent_date",
    "assignees.assignee_organization",
    "cpc_current.cpc_subgroup_id",
    "claims.claim_text",
]


class CheckpointError(ValueError):
    """Raised when a checkpoint file cannot be loaded."""


@dataclass
class Checkpoint:
    """Persisted pagination state for resumable ingest."""

    page: int = 1
    last_page: int = 0
    fetched_count: int = 0
    complete: bool = False
    fetched_ids: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointError(
                f"Corrupt checkpoint at {path}: invalid JSON ({exc})"
            ) from exc
        return cls(
            page=int(data.get("page", data.get("last_page", 0)) or 1),
            last_page=int(data.get("last_page", 0)),
            fetched_count=int(data.get("fetched_count", 0)),
            complete=bool(data.get("complete", False)),
            fetched_ids=list(data.get("fetched_ids", [])),
            records=list(data.get("records", [])),
        )


def _build_query(cpc_filter: str) -> dict[str, Any]:
    return {"_begins": {"cpc_subgroup_id": cpc_filter}}


def _request_with_backoff(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> dict[str, Any]:
    """POST with exponential backoff on transient errors (429, selected 5xx)."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.post(url, json=payload, headers=headers, timeout=60.0)
            if response.status_code in RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(
                    f"Retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in RETRYABLE_STATUS:
                raise
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc

        if attempt == max_retries - 1:
            break
        delay = base_delay * (2**attempt)
        logger.warning(
            "Request failed (attempt %s/%s): %s — retry in %.1fs",
            attempt + 1,
            max_retries,
            last_error,
            delay,
        )
        time.sleep(delay)
    assert last_error is not None
    raise last_error


def _first_claim_text(claims: Any) -> str:
    if not claims:
        return ""
    if isinstance(claims, list):
        if not claims:
            return ""
        first = claims[0]
        if isinstance(first, dict):
            return str(first.get("claim_text") or first.get("claim_text_raw") or "")
        return str(first)
    if isinstance(claims, dict):
        return str(claims.get("claim_text") or "")
    return str(claims)


def _extract_assignee(assignees: Any) -> str | None:
    if not assignees:
        return None
    names: list[str] = []
    if isinstance(assignees, list):
        for item in assignees:
            if isinstance(item, dict):
                org = item.get("assignee_organization") or item.get("assignee_name")
                if org:
                    names.append(str(org))
            elif item:
                names.append(str(item))
    elif isinstance(assignees, dict):
        org = assignees.get("assignee_organization") or assignees.get("assignee_name")
        if org:
            names.append(str(org))
    else:
        names.append(str(assignees))
    joined = "; ".join(names)
    return joined or None


def _extract_cpc_codes(cpc_current: Any) -> list[str]:
    if not cpc_current:
        return []
    codes: list[str] = []
    if isinstance(cpc_current, list):
        for item in cpc_current:
            if isinstance(item, dict):
                code = item.get("cpc_subgroup_id") or item.get("cpc_subgroup")
                if code:
                    codes.append(str(code))
            elif item:
                codes.append(str(item))
    elif isinstance(cpc_current, dict):
        code = cpc_current.get("cpc_subgroup_id") or cpc_current.get("cpc_subgroup")
        if code:
            codes.append(str(code))
    return codes


def parse_patent(raw: dict[str, Any]) -> PatentRecord:
    """Map PatentsView API record to PatentRecord."""
    return PatentRecord(
        patent_id=str(raw.get("patent_id") or ""),
        title=str(raw.get("patent_title") or ""),
        abstract=str(raw.get("patent_abstract") or ""),
        claim_snippet=_first_claim_text(raw.get("claims")),
        cpc_codes=_extract_cpc_codes(raw.get("cpc_current")),
        filing_date=raw.get("patent_date") or None,
        assignee=_extract_assignee(raw.get("assignees")),
    )


def try_parse_patent(raw: dict[str, Any]) -> PatentRecord | None:
    """Parse a patent record, skipping malformed API payloads."""
    try:
        return parse_patent(raw)
    except ValidationError as exc:
        patent_id = raw.get("patent_id", "unknown")
        logger.warning("Skipping invalid patent %s: %s", patent_id, exc)
        return None


def fetch_page(
    client: httpx.Client,
    settings: Settings,
    page: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch a single page; returns (patents, total_count)."""
    payload = {
        "q": _build_query(settings.cpc_filter),
        "f": PATENT_FIELDS,
        "o": {"page": page, "per_page": settings.per_page},
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.patentsview_api_key:
        headers["X-Api-Key"] = settings.patentsview_api_key

    data = _request_with_backoff(
        client, settings.patentsview_api_url, payload, headers
    )
    patents = data.get("patents") or []
    total = int(data.get("total_patent_count") or data.get("count") or 0)
    return patents, total


def _persist_record(checkpoint: Checkpoint, record: PatentRecord, path: Path) -> None:
    """Append record to checkpoint and save incrementally (per-record resume)."""
    if record.patent_id in checkpoint.fetched_ids:
        return
    checkpoint.fetched_ids.append(record.patent_id)
    checkpoint.records.append(record.model_dump(mode="json"))
    checkpoint.fetched_count = len(checkpoint.fetched_ids)
    checkpoint.save(path)


def fetch_patents(
    settings: Settings | None = None,
    *,
    limit: int | None = None,
    checkpoint_path: Path | None = None,
    on_page: Callable[[Checkpoint], None] | None = None,
    client: httpx.Client | None = None,
) -> Iterator[PatentRecord]:
    """
    Yield patents from PatentsView with checkpoint resume.

    Checkpoint stores fetched_ids and serialized records per patent. On failure,
    re-run to resume from the current page; already-fetched IDs are skipped.
    """
    settings = settings or get_settings()
    checkpoint_path = checkpoint_path or settings.checkpoint_path
    checkpoint = Checkpoint.load(checkpoint_path)
    seen_ids = set(checkpoint.fetched_ids)

    if checkpoint.complete:
        logger.info("Ingest already marked complete at checkpoint %s", checkpoint_path)
        return

    own_client = client is None
    http = client or httpx.Client()

    try:
        page = checkpoint.page
        fetched = checkpoint.fetched_count

        while True:
            if limit is not None and fetched >= limit:
                checkpoint.fetched_count = fetched
                checkpoint.complete = True
                checkpoint.save(checkpoint_path)
                break

            raw_patents, total = fetch_page(http, settings, page)
            if not raw_patents:
                checkpoint.complete = True
                checkpoint.save(checkpoint_path)
                break

            for raw in raw_patents:
                if limit is not None and fetched >= limit:
                    checkpoint.fetched_count = fetched
                    checkpoint.complete = True
                    checkpoint.save(checkpoint_path)
                    return

                record = try_parse_patent(raw)
                if not record or not record.patent_id:
                    continue
                if not record.abstract.strip():
                    logger.debug("Skipping patent %s: missing abstract", record.patent_id)
                    continue
                if record.patent_id in seen_ids:
                    continue

                _persist_record(checkpoint, record, checkpoint_path)
                seen_ids.add(record.patent_id)
                fetched += 1
                yield record

            page += 1
            checkpoint.page = page
            checkpoint.last_page = page - 1
            checkpoint.fetched_count = fetched
            checkpoint.save(checkpoint_path)
            if on_page:
                on_page(checkpoint)

            if fetched >= total:
                checkpoint.complete = True
                checkpoint.save(checkpoint_path)
                break
    finally:
        if own_client:
            http.close()