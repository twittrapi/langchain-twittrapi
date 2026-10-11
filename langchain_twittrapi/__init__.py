"""LangChain tools for TwittrAPI (https://twittrapi.com), a read API for X (Twitter) data.

Not affiliated with X Corp.
"""

from importlib import metadata

from langchain_twittrapi._output import DEFAULT_MAX_CHARS
from langchain_twittrapi.toolkits import TwittrAPIToolkit
from langchain_twittrapi.tools import (
    DEFAULT_API_RESPONSE_FORMAT,
    ApiResponseFormat,
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

try:
    __version__ = metadata.version("langchain-twittrapi")
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "__version__",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_API_RESPONSE_FORMAT",
    "ApiResponseFormat",
    "TwittrAPIToolkit",
    "TwittrAPIBaseTool",
    "GetUserByUsernameTool",
    "GetUserByIdTool",
    "GetUserTweetsTool",
    "GetFollowersTool",
    "GetFollowingTool",
    "SearchTweetsTool",
    "GetTweetTool",
    "GetTweetRetweetsTool",
    "GetTweetQuotesTool",
    "GetListTweetsTool",
    "GetCommunityTweetsTool",
    "GetTrendsTool",
    "GetTrendLocationsTool",
]
