"""
tests/test_summarizer.py

Tests for summarizer_node and its helper functions.

Run with:
    python -m pytest tests/test_summarizer.py -v
    -- or --
    python tests/test_summarizer.py   (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.summarizer import (
    _build_context,
    summarize_question,
    summarize_all,
    summarizer_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

QUESTION_1 = "How does drought caused by climate change reduce agricultural output?"
QUESTION_2 = "Which countries face the highest risk of famine due to rising temperatures?"

RESULTS_1 = [
    {
        "title": "Drought and Crop Failure: The Climate Connection",
        "url": "https://example.com/drought-crops",
        "content": (
            "Prolonged droughts reduce soil moisture by up to 40%, directly cutting "
            "wheat and maize yields. A 2023 FAO report found a 15% global reduction "
            "in crop productivity in drought-affected regions. Sub-Saharan Africa saw "
            "a 22% drop in cereal output between 2020-2023."
        ),
    },
    {
        "title": "Climate Change Impact on Global Agriculture",
        "url": "https://example.com/climate-ag",
        "content": (
            "Rising temperatures above 35C cause pollen sterility in rice crops, "
            "potentially reducing yields by 10-25% per degree of warming. Irrigation "
            "demand increases by 8-10% for every 1C rise in temperature."
        ),
    },
]

RESULTS_2 = [
    {
        "title": "Global Hunger Index 2023",
        "url": "https://example.com/hunger-index",
        "content": (
            "Somalia, South Sudan, and Yemen rank highest in the 2023 Global Hunger "
            "Index. Over 345 million people face acute food insecurity globally. "
            "Climate projections indicate a 30% increase in at-risk populations by 2050."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_summary(label: str, summary: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("="*60)
    print(summary)


# ---------------------------------------------------------------------------
# Test 1 -- _build_context helper (unit test, no API)
# ---------------------------------------------------------------------------

def test_build_context_no_api():
    """Verifies context builder concatenates and respects the char cap."""
    context = _build_context(RESULTS_1)

    assert isinstance(context, str), "Must return a string"
    assert len(context) <= 2000,     "Must respect _MAX_CONTEXT_CHARS"
    assert "Drought" in context,     "Must contain source content"
    assert "[" in context,           "Must include title brackets"

    print(f"\n[PASS] test_build_context_no_api -- context length: {len(context)} chars")


# ---------------------------------------------------------------------------
# Test 2 -- summarize_question (live Gemini call)
# ---------------------------------------------------------------------------

def test_summarize_question():
    """Verifies Gemini returns 3-5 bullet points for a single question."""
    summary = summarize_question(QUESTION_1, RESULTS_1)

    _print_summary("test_summarize_question", summary)

    assert isinstance(summary, str), "Must return a string"

    bullets = [ln for ln in summary.splitlines() if ln.strip()]
    assert 3 <= len(bullets) <= 5, (
        f"Expected 3-5 bullets, got {len(bullets)}: {bullets}"
    )
    for bullet in bullets:
        assert bullet.startswith("- "), f"Bullet must start with '- ': '{bullet}'"

    print("\n[PASS] test_summarize_question")


# ---------------------------------------------------------------------------
# Test 3 -- summarize_all (multiple questions)
# ---------------------------------------------------------------------------

def test_summarize_all():
    """Verifies parallel summaries list stays aligned with sub-questions."""
    summaries = summarize_all(
        [QUESTION_1, QUESTION_2],
        [RESULTS_1,  RESULTS_2],
    )

    for i, s in enumerate(summaries):
        _print_summary(f"test_summarize_all -- summary [{i+1}]", s)

    assert isinstance(summaries, list)
    assert len(summaries) == 2, f"Expected 2 summaries, got {len(summaries)}"
    for s in summaries:
        assert isinstance(s, str)
        assert len(s) > 0

    print("\n[PASS] test_summarize_all")


# ---------------------------------------------------------------------------
# Test 4 -- summarizer_node LangGraph interface
# ---------------------------------------------------------------------------

def test_summarizer_node():
    """Verifies the node reads correct keys and writes summaries."""
    state = {
        "query":              "Impact of climate change on food security",
        "sub_questions":      [QUESTION_1, QUESTION_2],
        "search_results":     [RESULTS_1, RESULTS_2],
        "summaries":          [],
        "consistency_report": "",
        "final_report":       "",
    }

    output = summarizer_node(state)

    _print_summary("test_summarizer_node -> summaries[0]", output["summaries"][0])
    _print_summary("test_summarizer_node -> summaries[1]", output["summaries"][1])

    assert "summaries" in output
    assert isinstance(output["summaries"], list)
    assert len(output["summaries"]) == 2

    print("\n[PASS] test_summarizer_node")


# ---------------------------------------------------------------------------
# Test 5 -- edge case: empty results returns fallback string (no API call)
# ---------------------------------------------------------------------------

def test_empty_results_fallback():
    """Verifies empty results return a safe fallback string."""
    summary = summarize_question(QUESTION_1, [])

    assert isinstance(summary, str)
    assert summary.startswith("- "), f"Fallback must start with '- ': '{summary}'"

    print(f"\n[PASS] test_empty_results_fallback -- got: '{summary}'")


# ---------------------------------------------------------------------------
# Test 6 -- edge case: length mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    """Verifies ValueError when sub_questions and search_results differ in length."""
    try:
        summarizer_node({
            "query":              "test",
            "sub_questions":      [QUESTION_1, QUESTION_2],
            "search_results":     [RESULTS_1],   # only 1 group for 2 questions
            "summaries":          [],
            "consistency_report": "",
            "final_report":       "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_length_mismatch_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running summarizer_node tests ]\n")
    test_build_context_no_api()          # no API key needed
    test_summarize_question()            # requires GEMINI_API_KEY
    test_summarize_all()                 # requires GEMINI_API_KEY
    test_summarizer_node()               # requires GEMINI_API_KEY
    test_empty_results_fallback()        # no API key needed
    test_length_mismatch_raises()        # no API key needed
    print("\n[ All tests passed ]\n")
