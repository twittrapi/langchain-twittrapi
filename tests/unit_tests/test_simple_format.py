"""Simple response format (``format=simple``), the default since 0.2.0.

Fixtures in tests/fixtures/simple are the API's simple-format output for real captured
responses, copied from the backend's normalize testdata. No network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from langchain_twittrapi import (
    GetFollowersTool,
    GetUserByUsernameTool,
    GetUserTweetsTool,
    SearchTweetsTool,
    TwittrAPIToolkit,
)
from langchain_twittrapi._output import CURSOR_HINT, LAST_PAGE, format_result
from tests.conftest import MockAPI, query

SIMPLE = Path(__file__).parent.parent / "fixtures" / "simple"


def simple(name: str) -> Any:
    return json.loads((SIMPLE / f"{name}.json").read_text(encoding="utf-8"))


def test_default_is_simple(api: MockAPI) -> None:
    tool = GetUserByUsernameTool(client=api.client())
    assert tool.api_response_format == "simple"
    tool.invoke({"username": "XDevelopers"})
    assert query(api.last) == {"format": "simple"}


def test_raw_opt_out(api: MockAPI) -> None:
    tool = GetUserTweetsTool(client=api.client(), api_response_format="raw")
    tool.invoke({"user_id": "12", "count": 5})
    assert query(api.last) == {"count": "5"}


def test_tool_setting_wins_over_client_default(api: MockAPI) -> None:
    # A client created with the SDK default (raw) still gets simple from the tool ...
    GetUserTweetsTool(client=api.client()).invoke({"user_id": "12"})
    assert query(api.last) == {"format": "simple"}
    # ... and a simple client gets raw when the tool asks for it.
    GetUserTweetsTool(client=api.client(response_format="simple"), api_response_format="raw").invoke(
        {"user_id": "12"}
    )
    assert query(api.last) == {}


def test_invalid_format_rejected(api: MockAPI) -> None:
    with pytest.raises(ValidationError):
        GetUserTweetsTool(client=api.client(), api_response_format="json")


async def test_async_sends_simple(api: MockAPI) -> None:
    api.respond(simple("user-userid-followers"))
    out = await GetFollowersTool(async_client=api.async_client()).ainvoke({"username": "XDevelopers"})
    assert query(api.last) == {"username": "XDevelopers", "format": "simple"}
    assert out.startswith("next_cursor: 1000000000000000000|")


def test_toolkit_default_and_raw(api: MockAPI) -> None:
    tools = TwittrAPIToolkit(client=api.client()).get_tools()
    assert {t.api_response_format for t in tools} == {"simple"}
    raw_tools = TwittrAPIToolkit(client=api.client(), api_response_format="raw").get_tools()
    assert {t.api_response_format for t in raw_tools} == {"raw"}
    search = [t for t in raw_tools if t.name == "search_tweets"][0]
    search.invoke({"query": "python"})
    assert query(api.last) == {"searchItem": "python"}


def test_simple_page_cursor_first(api: MockAPI) -> None:
    page = simple("user-userid-tweets")
    api.respond(page)
    out = GetUserTweetsTool(client=api.client()).invoke({"user_id": "2244994945"})
    lines = out.splitlines()
    assert lines[0] == f"next_cursor: {page['nextCursor']}"
    assert lines[1] == CURSOR_HINT
    body = json.loads("\n".join(lines[3:]))
    assert body["data"][0]["id"] == "2019881223666233717"
    assert body["hasNextPage"] is True


def test_simple_last_page(api: MockAPI) -> None:
    api.respond(simple("tweet-tweetid-favorites"))
    out = SearchTweetsTool(client=api.client()).invoke({"query": "nothing here"})
    assert out.splitlines()[0] == LAST_PAGE
    assert "Pass next_cursor" not in out


def test_has_next_page_false_wins_over_cursor() -> None:
    page = dict(simple("user-userid-tweets"), hasNextPage=False)
    assert format_result(page, paginated=True).splitlines()[0] == LAST_PAGE


def test_simple_cursor_survives_truncation() -> None:
    page = simple("user-userid-followers")
    out = format_result(page, max_chars=100, paginated=True)
    assert out.startswith(f"next_cursor: {page['nextCursor']}")
    assert "[TRUNCATED:" in out


def test_simple_profile_is_compact_json(api: MockAPI) -> None:
    profile = simple("user-by-username-username")
    api.respond(profile)
    out = GetUserByUsernameTool(client=api.client()).invoke({"username": "XDevelopers"})
    assert not out.startswith("next_cursor")
    assert json.loads(out) == profile
    assert '"username":"XDevelopers"' in out
