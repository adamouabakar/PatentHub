"""Pure validation helpers for the Streamlit UI (unit-testable)."""

from __future__ import annotations

EMPTY_QUERY_MESSAGE = "Entrez une recherche"
INDEX_DOWNLOAD_MESSAGE = "Impossible de charger l'index"


def is_search_disabled(status: str, release_url: str) -> bool:
    """
    Return True when the search button should be disabled.

    Search stays enabled for ``invalid`` when ``release_url`` is set so
    ``ensure_index()`` can remove a corrupt local directory and download.
    """
    if status == "missing":
        return True
    if status == "invalid":
        return not release_url
    return False


def validate_search_input(query: str, search_clicked: bool) -> str | None:
    """
    Return a user-facing validation message, or None if input is valid.

    When the user clicks search with empty/whitespace query, return FR message.
    """
    if search_clicked and not query.strip():
        return EMPTY_QUERY_MESSAGE
    return None