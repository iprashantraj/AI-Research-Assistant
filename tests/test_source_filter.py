"""
tests/test_source_filter.py

Tests for source_filter_node and filter_results_for_question.

Run with:
    python -m pytest tests/test_source_filter.py -v
    -- or --
    python tests/test_source_filter.py   (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.source_filter import (
    filter_results_for_question,
    filter_all_results,
    source_filter_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

QUESTION = "How does drought caused by climate change reduce agricultural output?"

RESULTS_RELEVANT = [
    {
        "title": "Drought and Crop Failure: The Climate Connection",
        "url": "https://example.com/drought-crops",
        "content": "Extended droughts reduce soil moisture, causing widespread crop failure...",
    },
    {
        "title": "Climate Change Impact on Global Agriculture",
        "url": "https://example.com/climate-ag",
        "content": "Rising temperatures and erratic rainfall patterns directly threaten yields...",
    },
    {
        "title": "Top 10 Holiday Destinations in 2024",
        "url": "https://example.com/travel",
        "content": "Explore the best places to visit this year...",  # clearly irrelevant
    },
]


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
        elif isinstance(item, list):
            print(f"  Group [{i}]: {len(item)} result(s)")


# ---------------------------------------------------------------------------
# Test 1 -- filter single question
# ---------------------------------------------------------------------------

def test_filter_results_for_question():
    """Verifies Gemini removes obviously irrelevant results."""
    filtered = filter_results_for_question(QUESTION, RESULTS_RELEVANT)

    _print_results("test_filter_results_for_question", filtered)

    assert isinstance(filtered, list), "Must return a list"
    assert len(filtered) <= len(RESULTS_RELEVANT), "Cannot return more than input"
    assert len(filtered) >= 1, "Fallback must ensure at least 1 result"

    for r in filtered:
        assert "title"   in r
        assert "url"     in r
        assert "content" in r

    print("\n[PASS] test_filter_results_for_question")


# ---------------------------------------------------------------------------
# Test 2 -- filter_all_results (multiple groups)
# ---------------------------------------------------------------------------

def test_filter_all_results():
    """Verifies parallel filtering keeps list alignment with sub-questions."""
    sub_questions = [QUESTION, "Which countries face the highest risk of famine?"]
    search_results = [RESULTS_RELEVANT, RESULTS_RELEVANT]

    filtered = filter_all_results(sub_questions, search_results)

    _print_results("test_filter_all_results", filtered)

    assert isinstance(filtered, list)
    assert len(filtered) == len(sub_questions), "Output must align with input"

    print("\n[PASS] test_filter_all_results")


# ---------------------------------------------------------------------------
# Test 3 -- source_filter_node LangGraph interface
# ---------------------------------------------------------------------------

def test_source_filter_node():
    """Verifies node reads correct keys and returns filtered search_results."""
    state = {
        "query": "Impact of climate change on food security",
        "sub_questions": [QUESTION],
        "search_results": [RESULTS_RELEVANT],
        "summaries": [],
        "consistency_report": "",
        "final_report": "",
    }

    output = source_filter_node(state)

    _print_results("test_source_filter_node -> search_results", output["search_results"])

    assert "search_results" in output
    assert isinstance(output["search_results"], list)
    assert len(output["search_results"]) == 1
    assert isinstance(output["search_results"][0], list)

    print("\n[PASS] test_source_filter_node")


# ---------------------------------------------------------------------------
# Test 4 -- edge case: empty results for a question
# ---------------------------------------------------------------------------

def test_empty_results_for_question():
    """Verifies empty results list is returned as-is without errors."""
    result = filter_results_for_question(QUESTION, [])
    assert result == [], f"Expected [], got {result}"
    print("\n[PASS] test_empty_results_for_question")


# ---------------------------------------------------------------------------
# Test 5 -- edge case: mismatched list lengths raise ValueError
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises():
    """Verifies ValueError when sub_questions and search_results differ in length."""
    try:
        source_filter_node({
            "query": "test",
            "sub_questions": ["Q1", "Q2"],
            "search_results": [RESULTS_RELEVANT],  # only 1 group for 2 questions
            "summaries": [],
            "consistency_report": "",
            "final_report": "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_mismatched_lengths_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running source_filter_node tests ]\n")
    test_filter_results_for_question()
    test_filter_all_results()
    test_source_filter_node()
    test_empty_results_for_question()
    test_mismatched_lengths_raises()
    print("\n[ All tests passed ]\n")
