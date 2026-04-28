"""
tests/test_searcher.py

Tests for searcher_node and the fetch_all_results / search_query helpers.

Run with:
    python -m pytest tests/test_searcher.py -v
    -- or --
    python tests/test_searcher.py   (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.search import search_query
from nodes.searcher import fetch_all_results, searcher_node


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_results(label: str, results: list):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("="*60)
    if not results:
        print("  (empty)")
        return
    for i, item in enumerate(results, 1):
        if isinstance(item, dict):
            print(f"  [{i}] {item.get('title', 'N/A')}")
            print(f"       URL     : {item.get('url', 'N/A')}")
            print(f"       Content : {item.get('content', '')[:80]}...")
        elif isinstance(item, list):
            print(f"  Sub-question [{i}]: {len(item)} result(s)")
            for j, r in enumerate(item, 1):
                print(f"    [{j}] {r.get('title', 'N/A')} | {r.get('url', 'N/A')}")


# ---------------------------------------------------------------------------
# Test 1 -- search_query helper (single query)
# ---------------------------------------------------------------------------

def test_search_query_single():
    """Verifies a single Tavily query returns well-formed result dicts."""
    results = search_query("What is quantum entanglement?", max_results=3)

    _print_results("test_search_query_single", results)

    assert isinstance(results, list), "Must return a list"
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"

    for r in results:
        assert "title"   in r, "Missing 'title' key"
        assert "url"     in r, "Missing 'url' key"
        assert "content" in r, "Missing 'content' key"
        assert isinstance(r["title"],   str)
        assert isinstance(r["url"],     str)
        assert isinstance(r["content"], str)
        assert len(r["content"]) <= 500, "Content exceeds 500-char trim"

    print("\n[PASS] test_search_query_single")


# ---------------------------------------------------------------------------
# Test 2 -- fetch_all_results (multiple sub-questions)
# ---------------------------------------------------------------------------

def test_fetch_all_results():
    """Verifies parallel results list is aligned with input sub-questions."""
    sub_questions = [
        "How does climate change affect crop yields?",
        "What regions are most vulnerable to food insecurity due to climate change?",
    ]

    results = fetch_all_results(sub_questions)

    _print_results("test_fetch_all_results", results)

    assert isinstance(results, list), "Must return a list"
    assert len(results) == len(sub_questions), (
        f"Expected {len(sub_questions)} result groups, got {len(results)}"
    )

    for group in results:
        assert isinstance(group, list), "Each group must be a list"

    print("\n[PASS] test_fetch_all_results")


# ---------------------------------------------------------------------------
# Test 3 -- searcher_node LangGraph interface
# ---------------------------------------------------------------------------

def test_searcher_node():
    """Verifies the node reads sub_questions and writes search_results."""
    state = {
        "query": "Impact of climate change on food security",
        "sub_questions": [
            "How does drought caused by climate change reduce agricultural output?",
            "Which countries face the highest risk of famine due to rising temperatures?",
        ],
        "search_results": [],
        "summaries": [],
        "consistency_report": "",
        "final_report": "",
    }

    output = searcher_node(state)

    _print_results("test_searcher_node -> search_results", output["search_results"])

    assert "search_results" in output
    assert isinstance(output["search_results"], list)
    assert len(output["search_results"]) == len(state["sub_questions"])

    print("\n[PASS] test_searcher_node")


# ---------------------------------------------------------------------------
# Test 4 -- edge case: empty query to search_query
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty():
    """Verifies that a blank query returns an empty list, not an exception."""
    results = search_query("   ")
    assert results == [], f"Expected [], got {results}"
    print("\n[PASS] test_empty_query_returns_empty")


# ---------------------------------------------------------------------------
# Test 5 -- edge case: empty sub_questions raises ValueError
# ---------------------------------------------------------------------------

def test_empty_sub_questions_raises():
    """Verifies searcher_node raises ValueError if sub_questions is empty."""
    try:
        searcher_node({
            "query": "test",
            "sub_questions": [],
            "search_results": [],
            "summaries": [],
            "consistency_report": "",
            "final_report": "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_empty_sub_questions_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running searcher_node tests ]\n")
    test_search_query_single()
    test_fetch_all_results()
    test_searcher_node()
    test_empty_query_returns_empty()
    test_empty_sub_questions_raises()
    print("\n[ All tests passed ]\n")
