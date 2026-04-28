"""
core/graph.py

Assembles the LangGraph StateGraph and compiles it into a runnable pipeline.

Full pipeline (all nodes wired):
    planner -> searcher -> source_filter -> summarizer -> consistency -> report
"""

from langgraph.graph import StateGraph, END

from core.state import ResearchState
from nodes.planner       import planner_node
from nodes.searcher      import searcher_node
from nodes.source_filter import source_filter_node
from nodes.summarizer    import summarizer_node
from nodes.consistency   import consistency_checker_node
from nodes.report        import report_generator_node


def build_graph() -> StateGraph:
    """
    Constructs and compiles the full research assistant pipeline.

    Returns:
        A compiled LangGraph runnable ready to invoke with an initial state.
    """
    graph = StateGraph(ResearchState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("planner",       planner_node)
    graph.add_node("searcher",      searcher_node)
    graph.add_node("source_filter", source_filter_node)
    graph.add_node("summarizer",    summarizer_node)
    graph.add_node("consistency",   consistency_checker_node)
    graph.add_node("report",        report_generator_node)

    # ── Edges (linear pipeline) ───────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner",       "searcher")
    graph.add_edge("searcher",      "source_filter")
    graph.add_edge("source_filter", "summarizer")
    graph.add_edge("summarizer",    "consistency")
    graph.add_edge("consistency",   "report")
    graph.add_edge("report",        END)

    return graph.compile()


# Singleton — import this directly in app.py
research_graph = build_graph()
