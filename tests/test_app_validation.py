"""Tests for Streamlit input validation helpers."""

from __future__ import annotations

from app.validation import EMPTY_QUERY_MESSAGE, is_search_disabled, validate_search_input


def test_empty_query_returns_french_message() -> None:
    assert validate_search_input("", search_clicked=True) == EMPTY_QUERY_MESSAGE
    assert validate_search_input("   ", search_clicked=True) == EMPTY_QUERY_MESSAGE


def test_valid_query_returns_none() -> None:
    assert validate_search_input("réseaux de neurones", search_clicked=True) is None


def test_no_click_returns_none_even_if_empty() -> None:
    assert validate_search_input("", search_clicked=False) is None


def test_is_search_disabled_missing() -> None:
    assert is_search_disabled("missing", "") is True
    assert is_search_disabled("missing", "https://example.test/release.tar.gz") is True


def test_is_search_disabled_invalid() -> None:
    assert is_search_disabled("invalid", "") is True
    assert is_search_disabled("invalid", "https://example.test/release.tar.gz") is False


def test_is_search_disabled_ok_and_remote() -> None:
    assert is_search_disabled("ok", "") is False
    assert is_search_disabled("remote", "https://example.test/release.tar.gz") is False