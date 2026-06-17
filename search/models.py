"""Pydantic models for patent records."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatentRecord(BaseModel):
    """Normalized patent document for indexing and search."""

    patent_id: str
    title: str = ""
    abstract: str = ""
    claim_snippet: str = ""
    cpc_codes: list[str] = Field(default_factory=list)
    filing_date: str = ""
    assignee: str = ""

    def embed_text(self) -> str:
        """Concatenate fields used for embedding."""
        parts = [self.title.strip(), self.abstract.strip(), self.claim_snippet.strip()]
        return "\n\n".join(p for p in parts if p)