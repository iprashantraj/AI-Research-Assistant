"""
nodes/summarizer.py

summarizer_node -- LangGraph Node #4

Responsibility:
    Combine all sub-questions and their filtered source content into a SINGLE
    batched Groq call that extracts detailed bullet points for ALL questions.
    This keeps total API calls to a minimum.

Input  (from state):
    state["sub_questions"]  ->  list[str]
    state["search_results"] ->  list[list[dict]]   (already filtered)

Output (to state):
    state["summaries"]      ->  list[str]
    Each summary is a markdown bullet-point string (one per sub-question).
"""

import re
from urllib.parse import urlparse
from utils.llm import generate_text
from core.state import ResearchState

# Max total characters of source content per sub-question block
_MAX_CONTEXT_CHARS = 4000

# Simple in-process cache — avoids duplicate API calls within a session
_CACHE: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SUMMARIZER_PROMPT = """\
You are a research analyst. For EACH of the sub-questions below, combine insights \
across their respective sources to extract the most important and detailed facts.

Rules:
- For each sub-question, output EXACTLY this header format: [QUESTION N] where N is the index (1, 2, 3...)
- IMMEDIATELY UNDER the header, provide a 1-2 line analytical paragraph that captures the overall trend or main takeaway from the sources. It should feel analytical, not merely descriptive, and must NOT repeat the content of the bullets below. Do NOT use labels like "Insight Summary:".
- AFTER the analytical paragraph, output 4 to 6 detailed bullet points starting with "- "
- Write synthesis-style insights: combine related facts from multiple sources into a single cohesive bullet. Avoid "one source per bullet" structures.
- Make each bullet slightly more explanatory (1-2 lines long), providing context alongside the fact.
- Avoid repetitive phrasing across bullets.
- Each bullet MUST end with a clickable markdown citation formatted EXACTLY like this: (Source: [domain.com](URL))
- Do NOT include opinions, general statements, or filler.
- Do NOT add overall intro text or conclusions.

Input Data:
{mega_context}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(results: list[dict]) -> str:
    """Concatenate source content blocks into a trimmed context string."""
    parts = []
    total = 0

    for r in results:
        title   = r.get("title",   "Source").strip()
        url     = r.get("url", "")
        content = r.get("content", "").strip()
        
        domain = "unknown"
        if url:
            try:
                domain = urlparse(url).netloc.replace("www.", "")
            except Exception:
                pass
                
        block   = f"[{title}] (Source: {domain} | URL: {url})\n{content}"

        if total + len(block) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total
            if remaining > 100:
                parts.append(block[:remaining])
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def _build_mega_context(sub_questions: list[str], search_results: list[list[dict]]) -> str:
    """Combines all questions and their sources into one context block."""
    blocks = []
    for i, (q, results) in enumerate(zip(sub_questions, search_results), 1):
        blocks.append(f"=== [QUESTION {i}]: {q} ===")
        if not results:
            blocks.append("No relevant sources available.")
        else:
            blocks.append(_build_context(results))
        blocks.append("\n")
    return "\n".join(blocks)


def parse_summaries(raw_text: str, expected_count: int) -> list[str]:
    """Splits the batched LLM output back into parallel summary strings."""
    parts = re.split(r"\[QUESTION \d+\]", raw_text)

    summaries = []
    for part in parts[1:]:  # Skip preamble before first [QUESTION N]
        part = part.strip()

        lines = part.splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("* ", "• ")):
                line = "- " + line[2:]
            cleaned.append(line)

        if not cleaned:
            summaries.append("- Summary unavailable: model failed to extract facts from the provided sources.")
        else:
            summaries.append("\n\n".join(cleaned))

    # Pad if LLM missed some sections
    while len(summaries) < expected_count:
        summaries.append("- Summary unavailable: model failed to generate response for this section.")

    return summaries[:expected_count]


def summarize_all(
    sub_questions: list[str],
    search_results: list[list[dict]],
) -> list[str]:
    """Summarize all sub-questions using a SINGLE batched Groq API call."""
    mega_context = _build_mega_context(sub_questions, search_results)
    prompt = _SUMMARIZER_PROMPT.format(mega_context=mega_context)

    print(f"[INFO] Groq call -> summarizer ({len(sub_questions)} questions batched)")
    try:
        raw_text = generate_text(prompt).strip()
    except Exception as exc:
        print(f"[ERROR] Summarizer error: {exc}")
        return ["- Summary unavailable due to an API error."] * len(sub_questions)

    return parse_summaries(raw_text, len(sub_questions))


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def summarizer_node(state: ResearchState) -> dict:
    sub_questions:  list[str]        = state.get("sub_questions",  [])
    search_results: list[list[dict]] = state.get("search_results", [])

    if not sub_questions:
        raise ValueError("summarizer_node: state['sub_questions'] is empty.")

    if not search_results:
        raise ValueError("summarizer_node: state['search_results'] is empty.")

    if len(sub_questions) != len(search_results):
        raise ValueError(
            f"summarizer_node: Length mismatch — "
            f"{len(sub_questions)} sub_questions vs {len(search_results)} result groups."
        )

    summaries = summarize_all(sub_questions, search_results)

    return {"summaries": summaries}
