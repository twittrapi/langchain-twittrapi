from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from langchain_twittrapi import TwittrAPIToolkit
from tests.conftest import MockAPI

ALL_TOOLS = [
    "get_user_by_username",
    "get_user_by_id",
    "get_user_tweets",
    "get_followers",
    "get_following",
    "search_tweets",
    "get_tweet",
    "get_tweet_retweets",
    "get_tweet_quotes",
    "get_list_tweets",
    "get_community_tweets",
    "get_trends",
    "get_trend_locations",
]


def test_lists_all_tools() -> None:
    tools = TwittrAPIToolkit(api_key="k").get_tools()
    assert [t.name for t in tools] == ALL_TOOLS
    assert all(isinstance(t, BaseTool) for t in tools)
    assert TwittrAPIToolkit.tool_names() == ALL_TOOLS


def test_tools_share_clients() -> None:
    toolkit = TwittrAPIToolkit(api_key="k")
    tools = toolkit.get_tools()
    assert {id(t.client) for t in tools} == {id(toolkit.client)}
    assert {id(t.async_client) for t in tools} == {id(toolkit.async_client)}


def test_auth_modes() -> None:
    direct = TwittrAPIToolkit(api_key="k")
    assert direct.client.auth_mode == direct.async_client.auth_mode == "direct"
    rapid = TwittrAPIToolkit(rapidapi_key="k")
    assert rapid.client.auth_mode == rapid.async_client.auth_mode == "rapidapi"


def test_both_keys_rejected() -> None:
    with pytest.raises(ValueError, match="not both"):
        TwittrAPIToolkit(api_key="a", rapidapi_key="b")


def test_missing_key() -> None:
    with pytest.raises(ValueError, match="No API key found"):
        TwittrAPIToolkit()


def test_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTRAPI_API_KEY", "env")
    assert TwittrAPIToolkit().client.auth_mode == "direct"


def test_subset_and_unknown() -> None:
    toolkit = TwittrAPIToolkit(api_key="k")
    assert [t.name for t in toolkit.get_tools(["get_tweet", "search_tweets"])] == ["search_tweets", "get_tweet"]
    with pytest.raises(ValueError, match="Unknown TwittrAPI tool"):
        toolkit.get_tools(["post_tweet"])


def test_max_chars_propagates() -> None:
    assert {t.max_chars for t in TwittrAPIToolkit(api_key="k", max_chars=500).get_tools()} == {500}


def test_custom_clients(api: MockAPI) -> None:
    toolkit = TwittrAPIToolkit(client=api.client())
    tool = toolkit.get_tools(["get_user_by_id"])[0]
    assert tool.invoke({"user_id": "5"}) == '{"ok":true}'
    assert api.last.url.path == "/user/5"


def test_rapidapi_host() -> None:
    toolkit = TwittrAPIToolkit(rapidapi_key="k", rapidapi_host="twttr.p.rapidapi.com")
    assert toolkit.client.rapidapi_host == toolkit.async_client.rapidapi_host == "twttr.p.rapidapi.com"
    assert toolkit.client.base_url == "https://twttr.p.rapidapi.com"
    assert TwittrAPIToolkit(rapidapi_key="k").client.rapidapi_host == "twitter-aio.p.rapidapi.com"
