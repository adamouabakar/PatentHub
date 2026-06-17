"""Tests for Streamlit input validation helpers."""

from __future__ import annotations

from app.validation import EMPTY_QUERY_MESSAGE, validate_search_input


def test_empty_query_returns_french_message() -> None:
    assert validate_search_input("", search_clicked=True) == EMPTY_QUERY_MESSAGE
    assert validate_search_input("   ", search_clicked=True) == EMPTY_QUERY_MESSAGE


def test_valid_query_returns_none() -> None:
    assert validate_search_input("réseaux de neurones", search_clicked=True) is None


def test_no_click_returns_none_even_if_empty() -> None:
    assert validate_search_input("", search_clicked=False) is None