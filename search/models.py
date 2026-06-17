"""Pydantic models for patent records."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class PatentRecord(BaseModel):
    """Normalized patent document for indexing and search."""

    patent_id: str
    title: str = ""
    abstract: str = ""
    claim_snippet: str = ""
    cpc_codes: list[str] = Field(default_factory=list)
    filing_date: date | None = None
    assignee: str | None = None

    @field_validator("filing_date", mode="before")
    @classmethod
    def _parse_filing_date(cls, value: object) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    def embed_text(self) -> str:
        """Concatenate fields used for embedding."""
        parts = [self.title.strip(), self.abstract.strip(), self.claim_snippet.strip()]
        return "\n\n".join(p for p in parts if p)

    def filing_date_str(self) -> str:
        """ISO date string for LanceDB storage."""
        return self.filing_date.isoformat() if self.filing_date else ""