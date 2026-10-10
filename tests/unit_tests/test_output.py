from langchain_twittrapi._output import DEFAULT_MAX_CHARS, format_result, truncate

from tests.conftest import CURSOR_PAGE


def test_default_max_chars() -> None:
    assert DEFAULT_MAX_CHARS == 20_000


def test_short_text_untouched() -> None:
    assert truncate("abc", 10) == "abc"


def test_truncation_marker() -> None:
    out = truncate("x" * 500, 100)
    assert out.startswith("x" * 100)
    assert "x" * 101 not in out
    assert "[TRUNCATED: showing the first 100 of 500 characters (400 cut)." in out


def test_non_paginated_is_plain_json() -> None:
    assert format_result({"a": 1}) == '{"a":1}'


def test_unicode_kept() -> None:
    assert format_result({"t": "héllo 👋"}) == '{"t":"héllo 👋"}'


def test_cursor_first_and_outside_budget() -> None:
    out = format_result(CURSOR_PAGE, max_chars=100, paginated=True)
    lines = out.splitlines()
    assert lines[0] == "next_cursor: NEXT_CURSOR_123"
    assert "Pass next_cursor as `cursor`" in lines[1]
    assert "[TRUNCATED:" in out


def test_missing_cursor_is_explicit() -> None:
    out = format_result({"data": {}}, paginated=True)
    assert out.startswith("next_cursor: none found")
