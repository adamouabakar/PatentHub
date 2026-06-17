"""Tests for embed text preparation."""

from __future__ import annotations

from search.models import PatentRecord
from ingest.embed import build_embed_text, truncate_to_tokens


def test_truncate_to_tokens_limits_words() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    truncated = truncate_to_tokens(text, max_tokens=512)
    assert len(truncated.split()) == 512


def test_truncate_empty_returns_empty() -> None:
    assert truncate_to_tokens("", max_tokens=512) == ""


def test_build_embed_text_concatenates_fields() -> None:
    record = PatentRecord(
        patent_id="1",
        title="AI System",
        abstract="Deep learning pipeline.",
        claim_snippet="1. A method comprising a neural network.",
    )
    text = build_embed_text(record, max_tokens=512)
    assert "AI System" in text
    assert "Deep learning" in text
    assert "neural network" in text


def test_build_embed_text_truncates_long_content() -> None:
    record = PatentRecord(
        patent_id="2",
        title="T",
        abstract=" ".join(["abstract"] * 400),
        claim_snippet=" ".join(["claim"] * 400),
    )
    text = build_embed_text(record, max_tokens=100)
    assert len(text.split()) <= 100


def test_patent_record_embed_text() -> None:
    record = PatentRecord(
        patent_id="3",
        title="Title",
        abstract="Abstract",
        claim_snippet="Claim",
    )
    assert record.embed_text() == "Title\n\nAbstract\n\nClaim"