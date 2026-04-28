"""
nodes/report.py

report_generator_node -- LangGraph Node #6 (Final)

Responsibility:
    Assemble a clean, structured markdown research report from the data
    already present in state. Does NOT generate new facts.

Input  (from state):
    state["query"]              ->  str
    state["sub_questions"]      ->  list[str]
    state["summaries"]          ->  list[str]
    state["consistency_report"] ->  str

Output (to state):
    state["final_report"]       ->  str  (full markdown document)

Architecture decision:
    All content (summaries, consistency report) is assembled in pure
    Python — this guarantees "no new facts" and keeps formatting deterministic.
    LLM is NO LONGER used here to save API calls.
"""

from core.state import ResearchState


def _assemble_report(
    title: str,
    sub_questions: list[str],
    summaries: list[str],
    consistency_report: str,
) -> str:
    """
    Assemble the final markdown report in pure Python.
    """
    sections: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    sections.append(f"# {title}\n")

    # ── Key Findings ─────────────────────────────────────────────────────────
    sections.append("## Key Findings\n")

    for question, summary in zip(sub_questions, summaries):
        safe_question = question.strip().replace("\n", " ")
        sections.append(f"### {safe_question}\n")
        # Summaries are bullet-point strings — insert verbatim with spacing
        sections.append(f"{summary.strip()}\n\n")

    # Handle mismatched lengths: if more questions than summaries, note missing
    if len(sub_questions) > len(summaries):
        extra_questions = sub_questions[len(summaries):]
        for question in extra_questions:
            safe_question = question.strip().replace("\n", " ")
            sections.append(f"### {safe_question}\n")
            sections.append("- Summary not available for this sub-question.\n\n")

    # ── Consistency Check ─────────────────────────────────────────────────────
    sections.append("## Consistency Check\n")
    sections.append(
        consistency_report.strip()
        if consistency_report.strip()
        else "_No consistency report available._"
    )

    return "\n".join(sections)


def build_report(
    query: str,
    sub_questions: list[str],
    summaries: list[str],
    consistency_report: str,
) -> str:
    print("[report] Formatting report (pure Python)...")
    title = query.strip().title()
    report = _assemble_report(title, sub_questions, summaries, consistency_report)
    print(f"[report] Report generated — {len(report)} chars, "
          f"{report.count(chr(10))+1} lines.")

    return report


def report_generator_node(state: ResearchState) -> dict:
    """
    LangGraph-compatible node function.
    """
    query:              str        = state.get("query",              "").strip()
    sub_questions:      list[str]  = state.get("sub_questions",      [])
    summaries:          list[str]  = state.get("summaries",          [])
    consistency_report: str        = state.get("consistency_report", "")

    if not query:
        raise ValueError("report_generator_node: state['query'] is empty.")

    if not sub_questions:
        raise ValueError("report_generator_node: state['sub_questions'] is empty.")

    if not summaries:
        raise ValueError("report_generator_node: state['summaries'] is empty.")

    final_report = build_report(query, sub_questions, summaries, consistency_report)

    return {"final_report": final_report}
