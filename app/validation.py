"""Pure validation helpers for the Streamlit UI (unit-testable)."""

from __future__ import annotations

EMPTY_QUERY_MESSAGE = "Entrez une recherche"
INDEX_DOWNLOAD_MESSAGE = "Impossible de charger l'index"


def validate_search_input(query: str, search_clicked: bool) -> str | None:
    """
    Return a user-facing validation message, or None if input is valid.

    When the user clicks search with empty/whitespace query, return FR message.
    """
    if search_clicked and not query.strip():
        return EMPTY_QUERY_MESSAGE
    return None