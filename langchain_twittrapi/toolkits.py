"""A toolkit that returns every TwittrAPI tool with one shared client."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, SecretStr
from twittrapi import AsyncTwittrAPI, TwittrAPI, TwittrAPIError

from ._output import DEFAULT_MAX_CHARS
from .tools import (
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

__all__ = ["TwittrAPIToolkit", "TOOL_CLASSES"]

TOOL_CLASSES: Sequence[Type[TwittrAPIBaseTool]] = (
    GetUserByUsernameTool,
    GetUserByIdTool,
    GetUserTweetsTool,
    GetFollowersTool,
    GetFollowingTool,
    SearchTweetsTool,
    GetTweetTool,
    GetTweetRetweetsTool,
    GetTweetQuotesTool,
    GetListTweetsTool,
    GetCommunityTweetsTool,
    GetTrendsTool,
    GetTrendLocationsTool,
)


def _secret(value: Optional[SecretStr]) -> Optional[str]:
    return value.get_secret_value() if value is not None else None


class TwittrAPIToolkit(BaseToolkit):
    """All TwittrAPI tools, sharing one sync and one async client.

    Setup:
        .. code-block:: bash

            pip install -U langchain-twittrapi
            export TWITTRAPI_API_KEY="..."   # or RAPIDAPI_KEY="..."

    Instantiate:
        .. code-block:: python

            from langchain_twittrapi import TwittrAPIToolkit

            toolkit = TwittrAPIToolkit()                      # key from the environment
            toolkit = TwittrAPIToolkit(api_key="...")         # direct API key
            toolkit = TwittrAPIToolkit(rapidapi_key="...")    # RapidAPI key

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()
            tools = toolkit.get_tools(["search_tweets", "get_tweet"])
    """

    api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    """Direct TwittrAPI key (https://twittrapi.com)."""
    rapidapi_key: Optional[SecretStr] = Field(default=None, exclude=True)
    """RapidAPI key."""
    base_url: Optional[str] = None
    """Override the API base URL. Rarely needed."""
    client: Any = Field(default=None, exclude=True)
    """A ``twittrapi.TwittrAPI`` instance. Built from the keys when not given."""
    async_client: Any = Field(default=None, exclude=True)
    """A ``twittrapi.AsyncTwittrAPI`` instance. Built from the keys when neither client is given."""
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=100)
    """Maximum characters of JSON each tool returns to the model."""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.client is None and self.async_client is None:
            kwargs = {
                "api_key": _secret(self.api_key),
                "rapidapi_key": _secret(self.rapidapi_key),
                "base_url": self.base_url,
            }
            try:
                self.client = TwittrAPI(**kwargs)
                self.async_client = AsyncTwittrAPI(**kwargs)
            except TwittrAPIError as exc:
                raise ValueError(str(exc)) from exc

    @staticmethod
    def tool_names() -> List[str]:
        """Names of all tools in the toolkit."""
        return [cls.model_fields["name"].default for cls in TOOL_CLASSES]

    def get_tools(self, names: Optional[Sequence[str]] = None) -> List[BaseTool]:
        """Returns the tools, optionally only those whose names are in ``names``."""
        if names is not None:
            unknown = set(names) - set(self.tool_names())
            if unknown:
                raise ValueError(
                    f"Unknown TwittrAPI tool(s): {', '.join(sorted(unknown))}. "
                    f"Available: {', '.join(self.tool_names())}"
                )
        return [
            cls(client=self.client, async_client=self.async_client, max_chars=self.max_chars)
            for cls in TOOL_CLASSES
            if names is None or cls.model_fields["name"].default in names
        ]
