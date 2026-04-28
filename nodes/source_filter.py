"""
nodes/source_filter.py

source_filter_node -- LangGraph Node #3

Responsibility:
    For each sub-question, filter the raw search results to remove duplicates
    and keep only the top 3 results.
    No LLM is used here to ensure maximum context is preserved for the summarizer.

Input  (from state):
    state["sub_questions"]  ->  list[str]
    state["search_results"] ->  list[list[dict]]

Output (to state):
    state["search_results"] ->  list[list[dict]]  (filtered in-place)
"""

from core.state import ResearchState

def filter_results_for_question(results: list[dict]) -> list[dict]:
    """
    Remove duplicates by URL and keep the top 3 results.
    """
    if not results:
        return []

    seen_urls = set()
    filtered = []
    
    for r in results:
        url = r.get("url", "").strip()
        if url and url in seen_urls:
            continue
        
        if url:
            seen_urls.add(url)
            
        filtered.append(r)
        
        if len(filtered) >= 3:
            break
            
    return filtered


def filter_all_results(
    sub_questions: list[str],
    search_results: list[list[dict]],
) -> list[list[dict]]:
    """
    Apply relevance filtering for every sub-question in parallel.
    """
    filtered_all: list[list[dict]] = []

    for question, results in zip(sub_questions, search_results):
        filtered = filter_results_for_question(results)
        print(f"[source_filter] '{question[:50]}...' kept {len(filtered)} unique results")
        filtered_all.append(filtered)

    return filtered_all


def source_filter_node(state: ResearchState) -> dict:
    """
    LangGraph-compatible node function.
    """
    sub_questions: list[str]       = state.get("sub_questions",  [])
    search_results: list[list[dict]] = state.get("search_results", [])

    if not sub_questions:
        raise ValueError("source_filter_node: state['sub_questions'] is empty.")

    if not search_results:
        raise ValueError("source_filter_node: state['search_results'] is empty.")

    if len(sub_questions) != len(search_results):
        raise ValueError(
            f"source_filter_node: Length mismatch — "
            f"{len(sub_questions)} sub_questions vs {len(search_results)} result groups."
        )

    filtered = filter_all_results(sub_questions, search_results)

    return {"search_results": filtered}
