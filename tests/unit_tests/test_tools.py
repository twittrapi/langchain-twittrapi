from __future__ import annotations

from typing import Any, Dict, Type

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_twittrapi import (
    GetCommunityTweetsTool,
    GetFollowersTool,
    GetFollowingTool,
    GetListTweetsTool,
    GetTrendLocationsTool,
    GetTrendsTool,
    GetTweetQuotesTool,
    GetTweetRetweetsTool,
    GetTweetTool,
    GetUserByIdTool,
    GetUserByUsernameTool,
    GetUserTweetsTool,
    SearchTweetsTool,
    TwittrAPIBaseTool,
)
from tests.conftest import CURSOR_PAGE, MockAPI, query

# tool class, invoke args, expected path, expected query params, paginated
CASES = [
    (GetUserByUsernameTool, {"username": "@jack"}, "/user/by/username/jack", {}, False),
    (GetUserByIdTool, {"user_id": "12"}, "/user/12", {}, False),
    (GetUserTweetsTool, {"user_id": "12", "count": 5}, "/user/12/tweets", {"count": "5"}, True),
    (GetUserTweetsTool, {"username": "jack"}, "/user/-1/tweets", {"username": "jack"}, True),
    (GetFollowersTool, {"user_id": "12", "cursor": "C1"}, "/user/12/followers", {"cursor": "C1"}, True),
    (GetFollowingTool, {"username": "@jack"}, "/user/-1/followings", {"username": "jack"}, True),
    (
        SearchTweetsTool,
        {"query": "from:jack lang:en", "category": "Latest", "count": 20},
        "/search",
        {"searchItem": "from:jack lang:en", "category": "Latest", "count": "20"},
        True,
    ),
    (GetTweetTool, {"tweet_id": "20", "ranking_mode": "Recency"}, "/tweet/20", {"rankingMode": "Recency"}, True),
    (GetTweetRetweetsTool, {"tweet_id": "20", "count": 3}, "/tweet/20/retweets", {"count": "3"}, True),
    (GetTweetQuotesTool, {"tweet_id": "20"}, "/tweet/20/quotes", {}, True),
    (GetListTweetsTool, {"list_id": "99", "cursor": "C2"}, "/lists/tweets", {"listId": "99", "cursor": "C2"}, True),
    (GetCommunityTweetsTool, {"community_id": "77"}, "/communities/77/tweets", {}, True),
    (GetTrendsTool, {"location_id": "23424977"}, "/trends/23424977", {}, False),
    (GetTrendsTool, {}, "/trends/1", {}, False),
    (GetTrendLocationsTool, {}, "/misc/locations", {}, False),
]


@pytest.mark.parametrize("cls,args,path,params,paginated", CASES)
def test_sync_argument_passing(
    api: MockAPI, cls: Type[TwittrAPIBaseTool], args: Dict[str, Any], path: str, params: dict, paginated: bool
) -> None:
    api.respond(CURSOR_PAGE)
    tool = cls(client=api.client())
    out = tool.invoke(args)
    assert api.last.method == "GET"
    assert api.last.url.path == path
    assert query(api.last) == params
    if paginated:
        assert out.startswith("next_cursor: NEXT_CURSOR_123")
    else:
        assert out.startswith("{")


@pytest.mark.parametrize("cls,args,path,params,paginated", CASES)
async def test_async_argument_passing(
    api: MockAPI, cls: Type[TwittrAPIBaseTool], args: Dict[str, Any], path: str, params: dict, paginated: bool
) -> None:
    api.respond(CURSOR_PAGE)
    tool = cls(async_client=api.async_client())
    assert tool.client is None
    out = await tool.ainvoke(args)
    assert api.last.url.path == path
    assert query(api.last) == params
    assert ("next_cursor: NEXT_CURSOR_123" in out) == paginated


async def test_async_falls_back_to_sync_client(api: MockAPI) -> None:
    tool = GetUserByIdTool(client=api.client())
    out = await tool.ainvoke({"user_id": "12"})
    assert api.last.url.path == "/user/12"
    assert out == '{"ok":true}'


def test_sync_call_without_sync_client(api: MockAPI) -> None:
    tool = GetUserByIdTool(async_client=api.async_client())
    assert "No synchronous TwittrAPI client" in tool.invoke({"user_id": "12"})


def test_auth_header_direct(api: MockAPI) -> None:
    GetUserByIdTool(client=api.client(api_key="direct-key")).invoke({"user_id": "1"})
    assert api.last.headers["X-API-KEY"] == "direct-key"
    assert "x-rapidapi-key" not in api.last.headers


def test_auth_header_rapidapi(api: MockAPI) -> None:
    GetUserByIdTool(client=api.client(api_key=None, rapidapi_key="rapid-key")).invoke({"user_id": "1"})
    assert api.last.headers["x-rapidapi-key"] == "rapid-key"
    assert api.last.url.host == "twitter-aio.p.rapidapi.com"


def test_builds_clients_from_keys() -> None:
    direct = GetTweetTool(api_key="k1")
    assert direct.client.auth_mode == "direct"
    assert direct._get_async_client().auth_mode == "direct"
    rapid = GetTweetTool(rapidapi_key="k2")
    assert rapid.client.auth_mode == "rapidapi"
    assert rapid._get_async_client().auth_mode == "rapidapi"


def test_builds_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPIDAPI_KEY", "env-key")
    assert GetTweetTool().client.auth_mode == "rapidapi"


def test_missing_key_is_clear() -> None:
    with pytest.raises(ValueError, match="TWITTRAPI_API_KEY"):
        GetTweetTool()


def test_keys_not_serialized() -> None:
    tool = GetTweetTool(api_key="secret-value")
    assert "secret-value" not in repr(tool)
    assert "api_key" not in tool.model_dump()


# --- schemas ----------------------------------------------------------------

EXPECTED_REQUIRED = {
    GetUserByUsernameTool: ["username"],
    GetUserByIdTool: ["user_id"],
    GetUserTweetsTool: [],
    GetFollowersTool: [],
    GetFollowingTool: [],
    SearchTweetsTool: ["query"],
    GetTweetTool: ["tweet_id"],
    GetTweetRetweetsTool: ["tweet_id"],
    GetTweetQuotesTool: ["tweet_id"],
    GetListTweetsTool: ["list_id"],
    GetCommunityTweetsTool: ["community_id"],
    GetTrendsTool: [],
    GetTrendLocationsTool: [],
}


@pytest.mark.parametrize("cls,required", list(EXPECTED_REQUIRED.items()))
def test_schema(api: MockAPI, cls: Type[TwittrAPIBaseTool], required: list) -> None:
    tool = cls(client=api.client())
    schema = tool.tool_call_schema.model_json_schema()
    assert schema.get("required", []) == required
    assert len(tool.description) > 20
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"{tool.name}.{name} has no description"


def test_schema_enums(api: MockAPI) -> None:
    props = SearchTweetsTool(client=api.client()).tool_call_schema.model_json_schema()["properties"]
    assert props["category"]["anyOf"][0]["enum"] == ["Top", "Latest", "People", "Photos", "Videos"]


@pytest.mark.parametrize(
    "cls,args",
    [
        (GetUserByIdTool, {"user_id": "abc"}),
        (GetTweetTool, {"tweet_id": "12x"}),
        (SearchTweetsTool, {"query": ""}),
        (SearchTweetsTool, {"query": "a", "category": "Nope"}),
        (GetUserTweetsTool, {"user_id": "1", "count": 0}),
        (GetUserTweetsTool, {"user_id": "1", "count": 101}),
    ],
)
def test_schema_rejects_bad_input(api: MockAPI, cls: Type[TwittrAPIBaseTool], args: dict) -> None:
    with pytest.raises(ValidationError):
        cls(client=api.client()).invoke(args)
    assert api.requests == []


# --- truncation ---------------------------------------------------------------


def test_truncates_to_max_chars(api: MockAPI) -> None:
    api.respond({"text": "y" * 5000})
    out = GetUserByIdTool(client=api.client(), max_chars=200).invoke({"user_id": "1"})
    assert "[TRUNCATED: showing the first 200 of" in out
    assert len(out) < 450


def test_cursor_survives_truncation(api: MockAPI) -> None:
    page = dict(CURSOR_PAGE, padding="z" * 5000)
    api.respond(page)
    out = GetUserTweetsTool(client=api.client(), max_chars=100).invoke({"user_id": "1"})
    assert out.startswith("next_cursor: NEXT_CURSOR_123")
    assert "[TRUNCATED:" in out


# --- errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,body,fragment",
    [
        (401, {"error": "Invalid API key"}, "Invalid API key"),
        (402, {"error": "Insufficient balance"}, "Insufficient balance"),
        (404, {"error": "User not found"}, "User not found"),
        (429, {"error": "Too many requests"}, "429"),
        (500, {"error": "boom"}, "500"),
    ],
)
def test_api_errors_returned_to_model(api: MockAPI, status: int, body: dict, fragment: str) -> None:
    api.respond(body, status)
    out = GetUserByIdTool(client=api.client()).invoke({"user_id": "1"})
    assert out.startswith("TwittrAPI error:")
    assert fragment in out


def test_api_error_as_tool_message(api: MockAPI) -> None:
    api.respond({"error": "Invalid API key"}, 401)
    msg = GetUserByIdTool(client=api.client()).invoke(
        {"type": "tool_call", "id": "call_1", "name": "get_user_by_id", "args": {"user_id": "1"}}
    )
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert "Invalid API key" in msg.content


async def test_async_api_error(api: MockAPI) -> None:
    api.respond({"error": "Not found"}, 404)
    out = await GetTweetTool(async_client=api.async_client()).ainvoke({"tweet_id": "1"})
    assert out.startswith("TwittrAPI error:") and "Not found" in out


def test_missing_user_target(api: MockAPI) -> None:
    out = GetFollowersTool(client=api.client()).invoke({})
    assert out == "Invalid arguments: Provide either user_id or username."
    assert api.requests == []


def test_errors_raise_when_handling_disabled(api: MockAPI) -> None:
    api.respond({"error": "Invalid API key"}, 401)
    tool = GetUserByIdTool(client=api.client(), handle_tool_error=False)
    with pytest.raises(ToolException, match="Invalid API key"):
        tool.invoke({"user_id": "1"})
