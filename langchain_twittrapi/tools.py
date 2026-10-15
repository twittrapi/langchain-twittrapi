"""LangChain tools for TwittrAPI (https://twittrapi.com), a read API for X (Twitter) data.

Every tool is read-only and calls the official ``twittrapi`` SDK. Results are the API's
compact simple format by default (flat, camelCase tweets and users, far fewer tokens than X's
raw JSON; https://twittrapi.com/docs/simple-format). Set ``api_response_format="raw"`` for X's
raw JSON. Output is cut to ``max_chars`` characters. Paginated tools print ``next_cursor``
first so the agent can ask for the next page.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Dict, Literal, Optional, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, PrivateAttr, SecretStr
from twittrapi import AsyncTwittrAPI, TwittrAPI, TwittrAPIError

from ._output import DEFAULT_MAX_CHARS, format_result

ApiResponseFormat = Literal["simple", "raw"]
DEFAULT_API_RESPONSE_FORMAT: ApiResponseFormat = "simple"

__all__ = [
    "ApiResponseFormat",
    "DEFAULT_API_RESPONSE_FORMAT",
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

# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------

_NUMERIC = r"^\d{1,30}$"

_CURSOR_DESC = (
    "Pagination cursor. Leave empty for the first page; pass the next_cursor from a previous "
    "result to get the next page."
)


def _count_field(what: str) -> Any:
    return Field(
        default=None,
        ge=1,
        le=100,
        description=f"How many {what} to request per page (default 20). X may return slightly more or fewer.",
    )


class UsernameInput(BaseModel):
    username: str = Field(
        min_length=1,
        max_length=20,
        description='X handle without the URL, e.g. "elonmusk" (a leading @ is fine).',
    )


class UserIdInput(BaseModel):
    user_id: str = Field(pattern=_NUMERIC, description="Numeric X user ID (rest_id).")


class UserTimelineInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        pattern=_NUMERIC,
        description="Numeric X user ID (rest_id). Provide this or username. Using the ID is faster.",
    )
    username: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="X handle, used when you do not have the user ID. Provide this or user_id.",
    )
    count: Optional[int] = _count_field("items")
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class SearchTweetsInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=512,
        description="Search query, max 512 characters. May contain X search operators.",
    )
    category: Optional[Literal["Top", "Latest", "People", "Photos", "Videos"]] = Field(
        default=None, description="Result tab: Top (default), Latest, People, Photos or Videos."
    )
    count: Optional[int] = _count_field("results")
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class TweetInput(BaseModel):
    tweet_id: str = Field(
        pattern=_NUMERIC, description="Numeric tweet ID, the number at the end of a tweet URL."
    )
    ranking_mode: Optional[Literal["Relevance", "Recency", "Likes"]] = Field(
        default=None, description="Sort order of the replies. Default Relevance."
    )
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class TweetPageInput(BaseModel):
    tweet_id: str = Field(pattern=_NUMERIC, description="Numeric tweet ID.")
    count: Optional[int] = _count_field("items")
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class ListTweetsInput(BaseModel):
    list_id: str = Field(pattern=_NUMERIC, description="Numeric list ID, from the list URL.")
    count: Optional[int] = _count_field("tweets")
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class CommunityTweetsInput(BaseModel):
    community_id: str = Field(pattern=_NUMERIC, description="Numeric community ID.")
    count: Optional[int] = _count_field("tweets")
    cursor: Optional[str] = Field(default=None, description=_CURSOR_DESC)


class TrendsInput(BaseModel):
    location_id: str = Field(
        default="1",
        pattern=r"^\d{1,12}$",
        description='Yahoo WOEID of the location, e.g. "1" for worldwide, "23424977" for the United States.',
    )


class NoInput(BaseModel):
    """This tool takes no arguments."""


# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------


def _secret(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return str(value)


def _client_kwargs(
    api_key: Any, rapidapi_key: Any, rapidapi_host: Optional[str], base_url: Optional[str]
) -> Dict[str, Any]:
    """Arguments for ``TwittrAPI`` / ``AsyncTwittrAPI``. ``rapidapi_host`` is only passed when
    set, so an unset host leaves the SDK's own default and ``RAPIDAPI_HOST`` fallback."""
    kwargs: Dict[str, Any] = {
        "api_key": _secret(api_key),
        "rapidapi_key": _secret(rapidapi_key),
        "base_url": base_url,
    }
    if rapidapi_host:
        kwargs["rapidapi_host"] = rapidapi_host
    return kwargs


class TwittrAPIBaseTool(BaseTool):
    """Shared plumbing for the TwittrAPI tools.

    Auth: pass ``api_key`` (direct API, https://twittrapi.com) or ``rapidapi_key`` (RapidAPI).
    With neither, the SDK reads ``TWITTRAPI_API_KEY`` and then ``RAPIDAPI_KEY`` from the
    environment. ``rapidapi_host`` (or ``RAPIDAPI_HOST``) selects another TwittrAPI listing on
    RapidAPI. You can also pass ready-made ``client`` (``twittrapi.TwittrAPI``) and
    ``async_client`` (``twittrapi.AsyncTwittrAPI``) instances.
    """

    api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    """Direct TwittrAPI key (sent as X-API-KEY)."""
    rapidapi_key: Optional[SecretStr] = Field(default=None, exclude=True)
    """RapidAPI key (sent as x-rapidapi-key)."""
    rapidapi_host: Optional[str] = None
    """RapidAPI host of the listing you subscribed to, e.g. ``twttr.p.rapidapi.com``. Defaults
    to ``RAPIDAPI_HOST`` from the environment, then ``twitter-aio.p.rapidapi.com``."""
    base_url: Optional[str] = None
    """Override the API base URL. Rarely needed."""
    client: Any = Field(default=None, exclude=True)
    """A ``twittrapi.TwittrAPI`` instance. Built from the keys when not given."""
    async_client: Any = Field(default=None, exclude=True)
    """A ``twittrapi.AsyncTwittrAPI`` instance. Built from the keys on first async call when not given."""
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=100)
    """Maximum characters of JSON returned to the model. Longer output is cut with a marker."""
    api_response_format: ApiResponseFormat = DEFAULT_API_RESPONSE_FORMAT
    """``"simple"`` (default): the API's compact format (``format=simple``), flat camelCase
    objects with ``nextCursor``/``hasNextPage``, far fewer tokens. ``"raw"``: X's raw GraphQL
    JSON with every field X sends. Sent on every call, whatever the client's own default.
    (Named ``api_response_format`` because LangChain's ``BaseTool`` already has a
    ``response_format`` field.)"""

    handle_tool_error: Any = True
    """API errors are returned to the model as the tool result instead of raising."""

    paginated: ClassVar[bool] = False

    _built_client: bool = PrivateAttr(default=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.client is None and self.async_client is None:
            try:
                self.client = TwittrAPI(
                    **_client_kwargs(self.api_key, self.rapidapi_key, self.rapidapi_host, self.base_url)
                )
            except TwittrAPIError as exc:
                raise ValueError(str(exc)) from exc
            # We built the sync client from the settings above, so the async client can be
            # built from the same settings on first use.
            self._built_client = True

    # Subclasses implement this. ``api`` is a TwittrAPI or AsyncTwittrAPI; the return value
    # is the parsed JSON for the sync client and a coroutine for the async one.
    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _format(self, data: Any) -> str:
        return format_result(data, max_chars=self.max_chars, paginated=self.paginated)

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        if self.client is None:
            raise ToolException("No synchronous TwittrAPI client is configured; call this tool asynchronously.")
        try:
            data = self._call(self.client, kwargs)
        except TwittrAPIError as exc:
            raise ToolException(f"TwittrAPI error: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise ToolException(f"Invalid arguments: {exc}") from exc
        return self._format(data)

    def _get_async_client(self) -> Optional[Any]:
        if self.async_client is None and self._built_client:
            self.async_client = AsyncTwittrAPI(
                **_client_kwargs(self.api_key, self.rapidapi_key, self.rapidapi_host, self.base_url)
            )
        return self.async_client

    async def _arun(
        self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: Any
    ) -> str:
        api = self._get_async_client()
        if api is None:
            # Only a sync client was supplied: run it in a worker thread.
            return await asyncio.get_running_loop().run_in_executor(None, lambda: self._run(**kwargs))
        try:
            data = await self._call(api, kwargs)
        except TwittrAPIError as exc:
            raise ToolException(f"TwittrAPI error: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise ToolException(f"Invalid arguments: {exc}") from exc
        return self._format(data)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _user_target(args: Dict[str, Any]) -> Dict[str, Any]:
    """Returns user_id/username for the SDK's user timeline methods (ID wins)."""
    user_id = args.get("user_id")
    username = args.get("username")
    if user_id:
        return {"user_id": user_id}
    if isinstance(username, str):
        handle = username.strip()
        handle = handle[1:].strip() if handle.startswith("@") else handle
        if handle:
            return {"username": handle}
    raise ValueError("Provide either user_id or username.")


class GetUserByUsernameTool(TwittrAPIBaseTool):
    """Get an X user's profile by handle."""

    name: str = "get_user_by_username"
    description: str = (
        "Get an X (Twitter) user's profile by handle: name, bio, follower/following counts, "
        "verification, created date, and the numeric user ID (rest_id) needed by other tools."
    )
    args_schema: Type[BaseModel] = UsernameInput

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.users.get_by_username(args["username"], response_format=self.api_response_format)


class GetUserByIdTool(TwittrAPIBaseTool):
    """Get an X user's profile by numeric ID."""

    name: str = "get_user_by_id"
    description: str = (
        "Get an X user's profile by numeric user ID (rest_id). Same data as get_user_by_username."
    )
    args_schema: Type[BaseModel] = UserIdInput

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.users.get_by_id(args["user_id"], response_format=self.api_response_format)


class GetUserTweetsTool(TwittrAPIBaseTool):
    """A user's recent tweets."""

    name: str = "get_user_tweets"
    description: str = (
        "Get a user's recent tweets (their profile timeline, without replies), newest first, "
        "including retweets and the pinned tweet. Takes user_id or username. Paginated."
    )
    args_schema: Type[BaseModel] = UserTimelineInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.users.tweets(
            **_user_target(args), count=args.get("count"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetFollowersTool(TwittrAPIBaseTool):
    """Accounts that follow a user."""

    name: str = "get_followers"
    description: str = (
        "Get accounts that follow a user, newest followers first. Takes user_id or username. Paginated."
    )
    args_schema: Type[BaseModel] = UserTimelineInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.users.followers(
            **_user_target(args), count=args.get("count"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetFollowingTool(TwittrAPIBaseTool):
    """Accounts a user follows."""

    name: str = "get_following"
    description: str = "Get accounts a user follows. Takes user_id or username. Paginated."
    args_schema: Type[BaseModel] = UserTimelineInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.users.following(
            **_user_target(args), count=args.get("count"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


# ---------------------------------------------------------------------------
# Search and tweets
# ---------------------------------------------------------------------------


class SearchTweetsTool(TwittrAPIBaseTool):
    """Search tweets like the X search box."""

    name: str = "search_tweets"
    description: str = (
        "Search X like the search box. Supports X search operators in the query, e.g. "
        'from:user, to:user, "exact phrase", lang:en, since:2025-01-01, until:2025-02-01, '
        "min_faves:100, filter:media, -filter:replies. category Latest gives newest first; "
        "People returns accounts instead of tweets. Paginated."
    )
    args_schema: Type[BaseModel] = SearchTweetsInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.search.tweets(
            args["query"],
            category=args.get("category"),
            count=args.get("count"),
            cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetTweetTool(TwittrAPIBaseTool):
    """A tweet with its reply thread."""

    name: str = "get_tweet"
    description: str = (
        "Get a tweet by ID with full text, author, media, engagement counts and the reply "
        "thread (conversation). Use cursor to load more replies."
    )
    args_schema: Type[BaseModel] = TweetInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.tweets.get(
            args["tweet_id"], ranking_mode=args.get("ranking_mode"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetTweetRetweetsTool(TwittrAPIBaseTool):
    """Users who retweeted a tweet."""

    name: str = "get_tweet_retweets"
    description: str = "Get the users who retweeted a tweet. Paginated."
    args_schema: Type[BaseModel] = TweetPageInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.tweets.retweeters(
            args["tweet_id"], count=args.get("count"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetTweetQuotesTool(TwittrAPIBaseTool):
    """Tweets that quote a tweet."""

    name: str = "get_tweet_quotes"
    description: str = "Get tweets that quote a given tweet. Paginated."
    args_schema: Type[BaseModel] = TweetPageInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.tweets.quotes(
            args["tweet_id"],
            count=args.get("count"),
            cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


# ---------------------------------------------------------------------------
# Lists, communities, trends
# ---------------------------------------------------------------------------


class GetListTweetsTool(TwittrAPIBaseTool):
    """Tweets from an X list."""

    name: str = "get_list_tweets"
    description: str = (
        "Get the tweet timeline of an X list (tweets from its members), newest first. Paginated."
    )
    args_schema: Type[BaseModel] = ListTweetsInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.lists.tweets(
            args["list_id"],
            count=args.get("count"),
            cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetCommunityTweetsTool(TwittrAPIBaseTool):
    """Tweets posted in an X Community."""

    name: str = "get_community_tweets"
    description: str = "Get tweets posted in an X Community. Paginated."
    args_schema: Type[BaseModel] = CommunityTweetsInput
    paginated: ClassVar[bool] = True

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.communities.tweets(
            args["community_id"], count=args.get("count"), cursor=args.get("cursor"),
            response_format=self.api_response_format,
        )


class GetTrendsTool(TwittrAPIBaseTool):
    """Trending topics for a location."""

    name: str = "get_trends"
    description: str = (
        "Get trending topics for a location by WOEID (1 = worldwide, 23424977 = United States, "
        "23424975 = United Kingdom). Use get_trend_locations to look up other WOEIDs."
    )
    args_schema: Type[BaseModel] = TrendsInput

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.trends.get(
            args.get("location_id") or "1",
            response_format=self.api_response_format,
        )


class GetTrendLocationsTool(TwittrAPIBaseTool):
    """Locations that have trends."""

    name: str = "get_trend_locations"
    description: str = (
        "List the locations (countries and cities) that have trends, with their WOEIDs for get_trends."
    )
    args_schema: Type[BaseModel] = NoInput

    def _call(self, api: Any, args: Dict[str, Any]) -> Any:
        return api.trends.locations(response_format=self.api_response_format)
