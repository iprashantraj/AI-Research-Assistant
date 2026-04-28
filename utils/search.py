"""
utils/search.py

Centralizes all Tavily API interactions.

IMPORTANT: TavilyClient is instantiated LAZILY (on first call to get_client()),
never at import time. This prevents crashes when the module is imported before
the .env file has been loaded.
"""

from __future__ import annotations
from tavily import TavilyClient
from utils.config import get_tavily_api_key

# Max characters to keep per result body — keeps downstream token usage low
_CONTENT_TRIM = 800

# Module-level cache — client is created once on first use, then reused
_client: TavilyClient | None = None


def get_client() -> TavilyClient:
    """
    Return a cached TavilyClient instance (lazy singleton).

    The client is created on the first call and reused for all subsequent
    calls. This avoids any initialization at import time.

    Returns:
        An authenticated TavilyClient.
    """
    global _client
    if _client is None:
        _client = TavilyClient(api_key=get_tavily_api_key())
    return _client


def search_query(
    query: str,
    max_results: int = 3,
) -> list[dict]:
    """
    Execute a single Tavily search and return normalized result dicts.

    Each returned dict is guaranteed to have the keys:
        - "title"   : str
        - "url"     : str
        - "content" : str  (trimmed to _CONTENT_TRIM chars)

    Args:
        query       : The search string to send to Tavily.
        max_results : Number of results to request (default: 3).

    Returns:
        A list of result dicts. Returns an empty list if Tavily returns
        nothing or raises an exception.
    """
    if not query or not query.strip():
        return []

    try:
        response = get_client().search(
            query=query.strip(),
            max_results=max_results,
            include_answer=False,   # raw results only — no LLM answer
            search_depth="basic",   # "basic" uses fewer API credits
        )
    except Exception as exc:
        print(f"[search] Tavily error for query '{query[:60]}': {exc}")
        return []

    raw_results: list[dict] = response.get("results", [])

    normalized = []
    for item in raw_results:
        normalized.append({
            "title":   str(item.get("title",   "")).strip(),
            "url":     str(item.get("url",     "")).strip(),
            "content": str(item.get("content", "")).strip()[:_CONTENT_TRIM],
        })

    return normalized
