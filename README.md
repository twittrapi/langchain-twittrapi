<p align="center">
  <a href="https://twittrapi.com"><img src=".github/banner.png" alt="langchain-twittrapi: Twitter/X tools for LangChain and LangGraph agents" width="100%"></a>
</p>

# Twitter/X tools for LangChain agents – TwittrAPI

[![PyPI](https://img.shields.io/pypi/v/langchain-twittrapi.svg)](https://pypi.org/project/langchain-twittrapi/)
[![test](https://github.com/twittrapi/langchain-twittrapi/actions/workflows/test.yml/badge.svg)](https://github.com/twittrapi/langchain-twittrapi/actions/workflows/test.yml)
[![Python](https://img.shields.io/pypi/pyversions/langchain-twittrapi.svg)](https://pypi.org/project/langchain-twittrapi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`langchain-twittrapi` is a set of LangChain tools for [TwittrAPI](https://twittrapi.com), a REST API for public Twitter/X data. LangChain and LangGraph agents can read tweets, profiles, followers, search results, lists, communities and trends. Pay as you go from $0.065 per 1,000 calls, no X developer account. Not affiliated with X Corp.

> **Get an API key:** [twittrapi.com/start](https://twittrapi.com/start?utm_source=github&utm_medium=readme&utm_campaign=langchain-twittrapi) · [Pricing](https://twittrapi.com/pricing?utm_source=github&utm_medium=readme&utm_campaign=langchain-twittrapi) · [Docs](https://twittrapi.com/docs?utm_source=github&utm_medium=readme&utm_campaign=langchain-twittrapi) · [Status](https://twittrapi.com/status) · also available on [RapidAPI](https://rapidapi.com/viperscores-viperscores-default/api/twitter-aio)

**SDKs and integrations:** [Python](https://github.com/twittrapi/twittrapi-python) · [Node.js](https://github.com/twittrapi/twittrapi-node) · [Go](https://github.com/twittrapi/twittrapi-go) · [MCP server](https://github.com/twittrapi/twittrapi-mcp) · [n8n](https://github.com/twittrapi/n8n-nodes-twittrapi) · [Make](https://github.com/twittrapi/twittrapi-make) · [Apify](https://github.com/twittrapi/twittrapi-apify) · [LlamaIndex](https://github.com/twittrapi/llama-index-tools-twittrapi) · [Agent skill](https://github.com/twittrapi/twittrapi-skills) · [Examples](https://github.com/twittrapi/twittrapi-examples)

All tools are read-only. They are built on the official [`twittrapi`](https://github.com/twittrapi/twittrapi-python) Python SDK.

## Install

```bash
pip install -U langchain-twittrapi
```

Python 3.10 or newer. Works with `langchain-core` 1.x.

## Authentication

Use either a direct TwittrAPI key or a RapidAPI key.

```bash
export TWITTRAPI_API_KEY="your-twittrapi-key"   # direct API
# or
export RAPIDAPI_KEY="your-rapidapi-key"         # RapidAPI
```

Or pass the key in code:

```python
from langchain_twittrapi import TwittrAPIToolkit

toolkit = TwittrAPIToolkit(api_key="your-twittrapi-key")     # direct API
toolkit = TwittrAPIToolkit(rapidapi_key="your-rapidapi-key") # RapidAPI
```

If both environment variables are set, `TWITTRAPI_API_KEY` wins.

## Quickstart

### Use one tool

```python
from langchain_twittrapi import GetUserByUsernameTool

tool = GetUserByUsernameTool()
print(tool.invoke({"username": "jack"}))
```

### Give an agent all the tools

```bash
pip install -U langchain "langchain[anthropic]"
```

```python
from langchain.agents import create_agent
from langchain_twittrapi import TwittrAPIToolkit

tools = TwittrAPIToolkit().get_tools()

agent = create_agent(
    "anthropic:claude-opus-5-5",
    tools=tools,
    system_prompt="You research X (Twitter) with the tools you have. Cite tweet IDs.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What has @jack posted this week? Summarize it."}]}
)
print(result["messages"][-1].content)
```

Any chat model that supports tool calling works. To pick a subset of tools:

```python
tools = TwittrAPIToolkit().get_tools(["search_tweets", "get_tweet"])
```

### Async

Every tool supports `ainvoke`. The toolkit creates one `AsyncTwittrAPI` client for async calls, so async agents do not block on HTTP.

```python
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Trending in the US?"}]})
```

## Tools

| Tool | Class | What it does |
| --- | --- | --- |
| `get_user_by_username` | `GetUserByUsernameTool` | Profile by handle, including the numeric user ID |
| `get_user_by_id` | `GetUserByIdTool` | Profile by numeric user ID |
| `get_user_tweets` | `GetUserTweetsTool` | A user's recent tweets (by `user_id` or `username`) |
| `get_followers` | `GetFollowersTool` | Accounts that follow a user |
| `get_following` | `GetFollowingTool` | Accounts a user follows |
| `search_tweets` | `SearchTweetsTool` | Search with X operators (`from:`, `lang:`, `since:`, `min_faves:` ...) |
| `get_tweet` | `GetTweetTool` | A tweet with its reply thread |
| `get_tweet_retweets` | `GetTweetRetweetsTool` | Users who retweeted a tweet |
| `get_tweet_quotes` | `GetTweetQuotesTool` | Tweets that quote a tweet |
| `get_list_tweets` | `GetListTweetsTool` | Tweets from an X list |
| `get_community_tweets` | `GetCommunityTweetsTool` | Tweets posted in an X Community |
| `get_trends` | `GetTrendsTool` | Trending topics by WOEID (1 = worldwide) |
| `get_trend_locations` | `GetTrendLocationsTool` | Locations that have trends, with their WOEIDs |

## What the tools return

The tools return the API's [simple format](https://twittrapi.com/docs/simple-format) as compact JSON text: flat, camelCase tweets and users with a fixed schema (`id`, `text`, `author`, `counts`, `media`, ...). It uses far fewer tokens than X's raw JSON. Lists come as `{"data": [...], "nextCursor": ..., "previousCursor": ..., "hasNextPage": ...}`, single objects as `{"data": {...}}`.

For X's raw JSON (every field X sends: cards, polls, edit history, ...), opt out per toolkit or per tool:

```python
TwittrAPIToolkit(api_response_format="raw")
GetTweetTool(api_response_format="raw")
```

(The option is named `api_response_format` because LangChain tools already have a `response_format` field.)

Responses can be large, so each result is cut at `max_chars` characters (default 20,000). When that happens the text ends with a `[TRUNCATED: ...]` marker that says how much was cut.

Paginated tools print the next-page cursor on the first line, before the JSON, so it is never lost to truncation:

```
next_cursor: DAABCgABF...
Pass next_cursor as `cursor` to get the next page.

{"data":[...],"nextCursor":"DAABCgABF...","previousCursor":"...","hasNextPage":true}
```

On the last page (`hasNextPage` is false) the first line says so instead.

Change the limit per toolkit or per tool:

```python
TwittrAPIToolkit(max_chars=8_000)
GetTweetTool(max_chars=50_000)
```

## Errors

API errors (bad key, no balance, not found, rate limit) and invalid arguments are returned to the model as the tool result, starting with `TwittrAPI error:` or `Invalid arguments:`. When the tool runs from a tool call, the `ToolMessage` has `status="error"`. To raise instead, pass `handle_tool_error=False`.

Requests that fail with 429, 5xx or a network error are retried by the SDK.

## Bring your own client

To change timeouts or retries, build the SDK clients yourself and pass them in:

```python
from twittrapi import AsyncTwittrAPI, TwittrAPI
from langchain_twittrapi import TwittrAPIToolkit

toolkit = TwittrAPIToolkit(
    client=TwittrAPI(api_key="...", timeout=30, max_retries=4),
    async_client=AsyncTwittrAPI(api_key="...", timeout=30, max_retries=4),
)
```

If you pass only `client`, async calls run it in a worker thread.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest pytest-asyncio langchain-tests
pytest
```

## Support

Questions, bugs or billing: [support@twittrapi.com](mailto:support@twittrapi.com). Issues with this package: [GitHub issues](https://github.com/twittrapi/langchain-twittrapi/issues).

## License

MIT. See [LICENSE](LICENSE).

Not affiliated with X Corp.
