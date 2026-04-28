"""
nodes/planner.py

planner_node -- LangGraph Node #1

Responsibility:
    Convert a vague user query into 3-5 focused, researchable sub-questions.

Input  (from state): state["query"]
Output (to   state): state["sub_questions"]  ->  list[str]
"""

import re
from utils.llm import generate_text
from core.state import ResearchState

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a research strategist. Break the following topic into 3 to 5 focused sub-questions \
that together cover the topic comprehensively.

Rules:
- Output ONLY a numbered list (1. 2. 3. ...).
- No intro text, no explanations, no blank lines between items.
- Each question must be specific and independently researchable.

Topic: {query}
"""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_sub_questions(query: str) -> list[str]:
    """
    Calls Groq and parses the response into a clean list of sub-questions.
    """
    prompt = _PLANNER_PROMPT.format(query=query.strip())

    print("[INFO] Groq call -> planner")
    raw_text: str = generate_text(prompt).strip()

    if raw_text.startswith("[ERROR]"):
        raise ValueError(f"planner_node: LLM call failed.\n{raw_text}")

    # Parse numbered list: "1. Question text" -> ["Question text", ...]
    lines = raw_text.splitlines()
    sub_questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1.", "1)", "1 -", etc.
        cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line)
        if cleaned:
            sub_questions.append(cleaned)

    if not sub_questions:
        raise ValueError(
            f"planner_node: LLM returned an unparseable response.\n"
            f"Raw output:\n{raw_text}"
        )

    return sub_questions[:5]  # Hard cap at 5 sub-questions


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def planner_node(state: ResearchState) -> dict:
    """
    LangGraph-compatible node function.
    Reads state["query"], writes state["sub_questions"].
    """
    query: str = state["query"]

    if not query or not query.strip():
        raise ValueError("planner_node: state['query'] is empty.")

    sub_questions = generate_sub_questions(query)

    return {"sub_questions": sub_questions}
