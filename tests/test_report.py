"""
tests/test_report.py

Tests for report_generator_node and its helper functions.

Run with:
    python -m pytest tests/test_report.py -v
    -- or --
    python tests/test_report.py   (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.report import (
    _generate_title,
    _assemble_report,
    build_report,
    report_generator_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

QUERY = "Impact of climate change on global food security"

SUB_QUESTIONS = [
    "How does drought caused by climate change reduce agricultural output?",
    "Which countries face the highest risk of famine due to rising temperatures?",
    "What adaptation strategies exist for climate-resilient agriculture?",
]

SUMMARIES = [
    (
        "- Droughts reduce soil moisture by up to 40%, cutting wheat and maize yields.\n"
        "- A 2023 FAO report found a 15% global drop in crop productivity in drought zones.\n"
        "- Sub-Saharan Africa saw a 22% drop in cereal output between 2020-2023."
    ),
    (
        "- Somalia, South Sudan, and Yemen rank highest on the 2023 Global Hunger Index.\n"
        "- Over 345 million people face acute food insecurity globally.\n"
        "- Climate projections indicate a 30% rise in at-risk populations by 2050."
    ),
    (
        "- Drought-resistant crop varieties maintain 70-80% of normal yields.\n"
        "- Precision irrigation reduces water use by up to 50%.\n"
        "- Agroforestry systems increase soil water retention by 20-30%."
    ),
]

CONSISTENCY_REPORT_CLEAN   = "No major conflicts found across sources."
CONSISTENCY_REPORT_CONFLICT = (
    "- CONFLICT: One source projects a 15% global yield increase; "
    "another projects a 25% decrease by 2050."
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_report(label: str, report: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("="*60)
    print(report[:600] + ("..." if len(report) > 600 else ""))


# ---------------------------------------------------------------------------
# Test 1 -- _assemble_report (pure Python, no API)
# ---------------------------------------------------------------------------

def test_assemble_report_no_api():
    """Verifies markdown structure without any API call."""
    report = _assemble_report(
        title="Climate Change and Global Food Security",
        sub_questions=SUB_QUESTIONS,
        summaries=SUMMARIES,
        consistency_report=CONSISTENCY_REPORT_CLEAN,
    )

    _print_report("test_assemble_report_no_api", report)

    assert "# Climate Change" in report,          "Must have H1 title"
    assert "## Key Findings"  in report,          "Must have Key Findings section"
    assert "## Consistency Check" in report,      "Must have Consistency Check section"
    assert SUB_QUESTIONS[0]   in report,          "Must embed first sub-question"
    assert "40%" in report,                       "Summaries must be verbatim"
    assert CONSISTENCY_REPORT_CLEAN in report,    "Must embed consistency report"

    # Verify section ordering: Key Findings must come before Consistency Check
    kf_pos = report.index("## Key Findings")
    cc_pos = report.index("## Consistency Check")
    assert kf_pos < cc_pos, "Key Findings must precede Consistency Check"

    print("\n[PASS] test_assemble_report_no_api")


# ---------------------------------------------------------------------------
# Test 2 -- summaries are verbatim (no API)
# ---------------------------------------------------------------------------

def test_summaries_are_verbatim_no_api():
    """Verifies every summary line appears exactly as-is in the report."""
    report = _assemble_report(
        title="Test Report",
        sub_questions=SUB_QUESTIONS,
        summaries=SUMMARIES,
        consistency_report=CONSISTENCY_REPORT_CLEAN,
    )

    for summary in SUMMARIES:
        for line in summary.splitlines():
            assert line.strip() in report, (
                f"Summary line missing from report: '{line}'"
            )

    print("\n[PASS] test_summaries_are_verbatim_no_api")


# ---------------------------------------------------------------------------
# Test 3 -- mismatched lengths handled safely (no API)
# ---------------------------------------------------------------------------

def test_mismatched_lengths_no_api():
    """Verifies report handles more sub-questions than summaries gracefully."""
    extra_questions = SUB_QUESTIONS + ["What is the role of governments in climate adaptation?"]
    report = _assemble_report(
        title="Test Report",
        sub_questions=extra_questions,
        summaries=SUMMARIES,   # only 3 summaries, 4 questions
        consistency_report=CONSISTENCY_REPORT_CLEAN,
    )

    _print_report("test_mismatched_lengths_no_api", report)

    assert "Summary not available for this sub-question." in report, (
        "Must handle missing summaries with a fallback note"
    )

    print("\n[PASS] test_mismatched_lengths_no_api")


# ---------------------------------------------------------------------------
# Test 4 -- _generate_title (live Gemini call)
# ---------------------------------------------------------------------------

def test_generate_title():
    """Verifies Gemini returns a non-empty title string."""
    title = _generate_title(QUERY)

    print(f"\n  Generated title: '{title}'")

    assert isinstance(title, str),  "Must return a string"
    assert len(title) > 0,          "Title must not be empty"
    assert len(title.split()) <= 12, f"Title too long: '{title}'"

    print("\n[PASS] test_generate_title")


# ---------------------------------------------------------------------------
# Test 5 -- build_report end-to-end (live Gemini call)
# ---------------------------------------------------------------------------

def test_build_report():
    """Verifies complete report assembly with live title generation."""
    report = build_report(QUERY, SUB_QUESTIONS, SUMMARIES, CONSISTENCY_REPORT_CLEAN)

    _print_report("test_build_report", report)

    assert report.startswith("#"),         "Report must start with an H1"
    assert "## Key Findings" in report,    "Must include Key Findings"
    assert "## Consistency Check" in report
    assert "40%" in report,               "Summaries must be verbatim"
    assert len(report) > 200,             "Report must have meaningful content"

    print("\n[PASS] test_build_report")


# ---------------------------------------------------------------------------
# Test 6 -- report_generator_node LangGraph interface (live Gemini call)
# ---------------------------------------------------------------------------

def test_report_generator_node():
    """Verifies the node reads all required keys and writes final_report."""
    state = {
        "query":              QUERY,
        "sub_questions":      SUB_QUESTIONS,
        "search_results":     [],
        "summaries":          SUMMARIES,
        "consistency_report": CONSISTENCY_REPORT_CONFLICT,
        "final_report":       "",
    }

    output = report_generator_node(state)

    _print_report("test_report_generator_node -> final_report",
                  output["final_report"])

    assert "final_report" in output
    assert isinstance(output["final_report"], str)
    assert output["final_report"].startswith("#")
    assert "CONFLICT" in output["final_report"], (
        "Consistency report with conflict must appear verbatim"
    )

    print("\n[PASS] test_report_generator_node")


# ---------------------------------------------------------------------------
# Test 7 -- edge case: empty query raises ValueError
# ---------------------------------------------------------------------------

def test_empty_query_raises():
    """Verifies ValueError is raised for a blank query."""
    try:
        report_generator_node({
            "query":              "",
            "sub_questions":      SUB_QUESTIONS,
            "search_results":     [],
            "summaries":          SUMMARIES,
            "consistency_report": "",
            "final_report":       "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_empty_query_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Test 8 -- edge case: empty summaries raises ValueError
# ---------------------------------------------------------------------------

def test_empty_summaries_raises():
    """Verifies ValueError is raised when summaries list is empty."""
    try:
        report_generator_node({
            "query":              QUERY,
            "sub_questions":      SUB_QUESTIONS,
            "search_results":     [],
            "summaries":          [],
            "consistency_report": "",
            "final_report":       "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_empty_summaries_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running report_generator_node tests ]\n")
    test_assemble_report_no_api()       # no API key needed
    test_summaries_are_verbatim_no_api()# no API key needed
    test_mismatched_lengths_no_api()    # no API key needed
    test_generate_title()               # requires GEMINI_API_KEY
    test_build_report()                 # requires GEMINI_API_KEY
    test_report_generator_node()        # requires GEMINI_API_KEY
    test_empty_query_raises()           # no API key needed
    test_empty_summaries_raises()       # no API key needed
    print("\n[ All tests passed ]\n")
