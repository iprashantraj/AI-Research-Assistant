"""
nodes/consistency.py

consistency_checker_node -- LangGraph Node #5

Responsibility:
    Optionally analyze all summaries and detect contradictions.
    Skipped by default (use_consistency=False) to save API calls.

Input  (from state):
    state["sub_questions"]    ->  list[str]
    state["summaries"]        ->  list[str]
    state["use_consistency"]  ->  bool (default: False)

Output (to state):
    state["consistency_report"] ->  str
"""

from utils.llm import generate_text
from core.state import ResearchState

# Cap on total characters of summaries fed to the LLM
_MAX_INPUT_CHARS = 3000

# Sentinels
_FALLBACK    = "No summaries available to check."
NO_CONFLICTS = "No major conflicts found across sources."

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_CONSISTENCY_PROMPT = """\
You are a fact-checking assistant. Review the research summaries below and \
identify any direct contradictions or conflicting claims between them.

Rules:
- If no contradictions exist, output exactly: No major conflicts found across sources.
- If contradictions exist, list each one as a bullet starting with "- CONFLICT:"
- Each bullet must be 1-2 lines maximum
- Output at most 5 conflict bullets
- Do NOT rewrite summaries
- Do NOT add explanations, headings, or intro text

Summaries:
{summaries_block}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summaries_block(sub_questions: list[str], summaries: list[str]) -> str:
    parts = []
    for i, (question, summary) in enumerate(zip(sub_questions, summaries), 1):
        block = f"[Q{i}] {question.strip()}\n{summary.strip()}"
        parts.append(block)

    combined = "\n\n".join(parts)
    return combined[:_MAX_INPUT_CHARS]


def check_consistency(sub_questions: list[str], summaries: list[str]) -> str:
    """Use Groq to cross-check summaries for contradictions."""
    if not summaries or all(not s.strip() for s in summaries):
        return _FALLBACK

    summaries_block = _build_summaries_block(sub_questions, summaries)
    prompt = _CONSISTENCY_PROMPT.format(summaries_block=summaries_block)

    try:
        print("[INFO] Groq call -> consistency checker")
        report = generate_text(prompt).strip()
        if report.startswith("[ERROR]"):
            print(f"[WARN] Consistency LLM error: {report}")
            return "Consistency check unavailable due to an API error."
    except Exception as exc:
        print(f"[ERROR] Consistency unexpected error: {exc}")
        return "Consistency check unavailable due to an API error."

    if "no major conflict" in report.lower():
        return NO_CONFLICTS

    # Enforce max 6 lines
    lines  = [ln for ln in report.splitlines() if ln.strip()]
    return "\n".join(lines[:6])


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def consistency_checker_node(state: ResearchState) -> dict:
    """
    LangGraph-compatible node function.
    Skips LLM call unless state["use_consistency"] is True.
    """
    sub_questions:   list[str] = state.get("sub_questions",   [])
    summaries:       list[str] = state.get("summaries",       [])
    use_consistency: bool      = state.get("use_consistency",  False)

    if not summaries:
        raise ValueError(
            "consistency_checker_node: state['summaries'] is empty. "
            "Ensure summarizer_node runs before consistency_checker_node."
        )

    if not use_consistency:
        print("[consistency] Skipping LLM check (use_consistency=False).")
        return {"consistency_report": NO_CONFLICTS}

    print(f"[consistency] Checking {len(summaries)} summaries for conflicts...")
    consistency_report = check_consistency(sub_questions, summaries)
    print(f"[consistency] Report: {consistency_report[:80]}...")

    return {"consistency_report": consistency_report}
