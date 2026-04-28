"""
nodes/searcher.py

searcher_node -- LangGraph Node #2

Responsibility:
    For every sub-question produced by the planner, execute a Tavily web
    search and store the top-3 results.

Input  (from state): state["sub_questions"]   ->  list[str]
Output (to   state): state["search_results"]  ->  list[list[dict]]

Result structure per sub-question:
    [
        {"title": str, "url": str, "content": str},
        ...  (up to 3 items)
    ]

No LLM is used here. All intelligence is deferred to downstream nodes.
"""

from utils.search import search_query
from core.state import ResearchState


# ---------------------------------------------------------------------------
# Core logic -- pure function, easy to unit-test
# ---------------------------------------------------------------------------

def fetch_all_results(sub_questions: list[str]) -> list[list[dict]]:
    """
    Run a Tavily search for each sub-question and collect results.

    Args:
        sub_questions: The list of focused sub-questions from the planner.

    Returns:
        A parallel list where index i contains the search results for
        sub_questions[i]. Empty inner list if a search returns nothing.
    """
    all_results: list[list[dict]] = []

    for question in sub_questions:
        results = search_query(question, max_results=3)
        all_results.append(results)

        status = f"{len(results)} result(s)" if results else "no results"
        print(f"[searcher] '{question[:60]}...' -> {status}")

    return all_results


# ---------------------------------------------------------------------------
# LangGraph node -- must accept and return state dict
# ---------------------------------------------------------------------------

def searcher_node(state: ResearchState) -> dict:
    """
    LangGraph-compatible node function.

    Reads  state["sub_questions"]
    Writes state["search_results"]

    Args:
        state: The current ResearchState.

    Returns:
        Partial state dict with "search_results" populated.

    Raises:
        ValueError: If sub_questions is missing or empty.
    """
    sub_questions: list[str] = state.get("sub_questions", [])

    if not sub_questions:
        raise ValueError(
            "searcher_node: state['sub_questions'] is empty. "
            "Ensure planner_node runs before searcher_node."
        )

    search_results = fetch_all_results(sub_questions)

    return {"search_results": search_results}
