from __future__ import annotations

from typing import Any, Type

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_twittrapi import GetUserByUsernameTool, SearchTweetsTool


class TestGetUserByUsernameStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[GetUserByUsernameTool]:
        return GetUserByUsernameTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"username": "jack"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"TWITTRAPI_API_KEY": "env-key"}, {}, {"max_chars": 20_000}


class TestSearchTweetsStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[SearchTweetsTool]:
        return SearchTweetsTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"rapidapi_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "from:jack", "category": "Latest", "count": 10}
