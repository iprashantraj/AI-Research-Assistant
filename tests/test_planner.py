"""
tests/test_planner.py

Tests for planner_node and the generate_sub_questions helper.

Run with:
    python -m pytest tests/test_planner.py -v
    -- or --
    python tests/test_planner.py   (standalone)
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.planner import generate_sub_questions, planner_node


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_result(label: str, result):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print("="*55)
    if isinstance(result, list):
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item}")
    else:
        print(f"  {result}")


# ---------------------------------------------------------------------------
# Test 1 -- core logic function
# ---------------------------------------------------------------------------

def test_generate_sub_questions():
    """Verifies the pure function returns a well-formed list of strings."""
    query = "What is the impact of climate change on global food security?"
    result = generate_sub_questions(query)

    _print_result("test_generate_sub_questions", result)

    assert isinstance(result, list), "Result should be a list"
    assert 3 <= len(result) <= 5, f"Expected 3-5 sub-questions, got {len(result)}"
    for q in result:
        assert isinstance(q, str), "Each sub-question must be a string"
        assert len(q) > 10, f"Sub-question too short: '{q}'"
        assert not q.strip()[0].isdigit(), f"Numbering not stripped: '{q}'"

    print("\n[PASS] test_generate_sub_questions")


# ---------------------------------------------------------------------------
# Test 2 -- LangGraph node interface
# ---------------------------------------------------------------------------

def test_planner_node():
    """Verifies the node function reads state['query'] and writes sub_questions."""
    state = {
        "query": "How does quantum computing threaten current encryption standards?",
        "sub_questions": [],
        "search_results": [],
        "summaries": [],
        "consistency_report": "",
        "final_report": "",
    }

    output = planner_node(state)

    _print_result("test_planner_node -> output dict", output)

    assert "sub_questions" in output, "Output must contain 'sub_questions'"
    assert isinstance(output["sub_questions"], list)
    assert 3 <= len(output["sub_questions"]) <= 5

    print("\n[PASS] test_planner_node")


# ---------------------------------------------------------------------------
# Test 3 -- edge case: empty query
# ---------------------------------------------------------------------------

def test_empty_query_raises():
    """Verifies a ValueError is raised for a blank query."""
    try:
        planner_node({
            "query": "   ",
            "sub_questions": [],
            "search_results": [],
            "summaries": [],
            "consistency_report": "",
            "final_report": "",
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] test_empty_query_raises -- caught: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[ Running planner_node tests ]\n")
    test_generate_sub_questions()
    test_planner_node()
    test_empty_query_raises()
    print("\n[ All tests passed ]\n")
