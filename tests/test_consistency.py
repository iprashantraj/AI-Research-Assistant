"""
tests/test_consistency.py

Tests for consistency_checker_node and its helper functions.

Run with:
    python -m pytest tests/test_consistency.py -v
    -- or --
    python tests/test_consistency.py   (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.consistency import (
    _build_summaries_block,
    check_consistency,
    consistency_checker_node,
    NO_CONFLICTS,
    _MAX_INPUT_CHARS,
    _FALLBACK,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

QUESTIONS = [
    "How does drought caused by climate change reduce agricultural output?",
    "Which countries face the highest risk of famine due to rising temperatures?",
    "What adaptation strategies exist for climate-resilient agriculture?",
]

# Summaries with NO contradictions
SUMMARIES_CLEAN = [
    (
        "- Droughts reduce soil moisture by up to 40%, cutting wheat and maize yields.\n"
        "- A 2023 FAO report found a 15% global drop in crop productivity in drought zones.\n"
        "- Sub-Saharan Africa saw a 22% drop in cereal output between 2020-2023.\n"
        "- Irrigation demand rises 8-10% per 1C temperature increase."
    ),
    (
        "- Somalia, South Sudan, and Yemen rank highest on the 2023 Global Hunger Index.\n"
        "- Over 345 million people face acute food insecurity globally.\n"
        "- Climate projections indicate a 30% rise in at-risk populations by 2050."
    ),
    (
        "- Drought-resistant crop varieties can maintain 70-80% of normal yields.\n"
        "- Precision irrigation reduces water use by up to 50%.\n"
        "- Agroforestry systems increase soil water retention by 20-30%."
    ),
]

# Summaries WITH a clear contradiction (yield reduction claim conflicts)
SUMMARIES_WITH_CONFLICT = [
    (
        "- Climate change is projected to increase global crop yields by 15% by 2050.\n"
        "- Improved seed varieties and CO2 fertilization may boost productivity.\n"
        "- Some temperate regions see modest gains from warming temperatures."
    ),
    (
        "- Climate change is expected to reduce global crop yields by up to 25% by 2050.\n"
        "- Over 345 million people already face acute food insecurity.\n"
        "- Extreme heat events are destroying harvests in tropical regions."
    ),
    (
        "- Adaptation strategies can offset some but not all productivity losses.\n"
        "- Net impact on yields remains contested between optimistic and pessimistic models.\n"
        "- Regional variation is high, making global averages misleading."
    ),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_report(label: str, report: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("="*60)
    print(report)


# ---------------------------------------------------------------------------
# Test 1 -- _build_summaries_block (unit test, no API)
# ---------------------------------------------------------------------------

def test_build_summaries_block_no_api():
    """Verifies block builder labels summaries and respects char cap."""
    block = _build_summaries_block(QUESTIONS, SUMMARIES_CLEAN)

    assert isinstance(block, str),         "Must return a string"
    assert len(block) <= _MAX_INPUT_CHARS, "Must respect _MAX_INPUT_CHARS"
    assert "[Q1]" in block,                "Must label first question"
    assert "[Q2]" in block,                "Must label second question"
    assert "drought" in block.lower(),     "Must include summary content"

    print(f"\n[PASS] test_build_summaries_block_no_api -- block length: {len(block)} chars")


# ---------------------------------------------------------------------------
# Test 2 -- check_consistency: no conflicts (live Gemini call)
# ---------------------------------------------------------------------------

def test_check_consistency_no_conflicts():
    """Verifies Gemini returns the no-conflict sentinel for consistent summaries."""
    report = check_consistency(QUESTIONS, SUMMARIES_CLEAN)

    _print_report("test_check_consistency_no_conflicts", report)

    assert isinstance(report, str),    "Must return a string"
    assert len(report) > 0,            "Must not be empty"
    # Should be the clean sentinel or a close variant
    assert "no major conflict" in report.lower(), (
        f"Expected 'no major conflict' in report, got: '{report}'"
    )

    print("\n[PASS] test_check_consistency_no_conflicts")


# ---------------------------------------------------------------------------
# Test 3 -- check_consistency: conflicts detected (live Gemini call)
# ---------------------------------------------------------------------------

def test_check_consistency_with_conflicts():
    """Verifies Gemini detects the explicit yield contradiction."""
    report = check_consistency(QUESTIONS, SUMMARIES_WITH_CONFLICT)

    _print_report("test_check_consistency_with_conflicts", report)

    assert isinstance(report, str), "Must return a string"
    lines = [ln for ln in report.splitlines() if ln.strip()]
    assert len(lines) <= 6, f"Must not exceed 6 lines, got {len(lines)}"

    # If conflicts found, at least one line should flag it
    if report != NO_CONFLICTS:
        has_conflict_marker = any(
            "conflict" in ln.lower() or "contradict" in ln.lower()
            for ln in lines
        )
        assert has_conflict_marker, (
            f"Expected conflict markers in report:\n{report}"
        )

    print("\n[PASS] test_check_consistency_with_conflicts")


# ---------------------------------------------------------------------------
# Test 4 -- consistency_checker_node LangGraph interface
# ---------------------------------------------------------------------------

def test_consistency_checker_node():
    """Verifies the node reads correct keys and writes consistency_report."""
    state = {
        "query":              "Impact of climate change on food security",
        "sub_questions":      QUESTIONS,
        "search_results":     [],
        "summaries":          SUMMARIES_CLEAN,
        "consistency_report": "",
        "final_report":       "",
    }

    output = consistency_checker_node(state)

    _print_report("test_consistency_checker_node -> consistency_report",
                  output["consistency_report"])

    assert "consistency_report" in output
    assert isinstance(output["consistency_report"], str)
    assert len(output["consistency_report"]) > 0

    print("\n[PASS] test_consistency_checker_node")


# ---------------------------------------------------------------------------
# Test 5 -- edge case: empty summaries returns fallback (no API)
# ---------------------------------------------------------------------------

def test_empty_summaries_fallback():
    """Verifies empty summaries return the fallback string without API call."""
    report = check_consistency([], [])

    assert report == _FALLBACK, f"Expected fallback, got: '{report}'"
    print(f"\n[PASS] test_empty_summaries_fallback -- got: '{report}'")


# ---------------------------------------------------------------------------
# Test 6 -- edge case: empty summaries raises ValueError via node
# ---------------------------------------------------------------------------

def test_empty_summaries_raises_in_node():
    """Verifies consistency_checker_node raises ValueError for empty summaries."""
    try:
        consistency_checker_node({
            "query":              "test",
            "sub_questions":      QUESTIONS,
            "search_results":     [],
            "summaries":          [],
            "consistency_report": "",
            "final_report":       "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_empty_summaries_raises_in_node -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running consistency_checker_node tests ]\n")
    test_build_summaries_block_no_api()      # no API key needed
    test_check_consistency_no_conflicts()    # requires GEMINI_API_KEY
    test_check_consistency_with_conflicts()  # requires GEMINI_API_KEY
    test_consistency_checker_node()          # requires GEMINI_API_KEY
    test_empty_summaries_fallback()          # no API key needed
    test_empty_summaries_raises_in_node()    # no API key needed
    print("\n[ All tests passed ]\n")
