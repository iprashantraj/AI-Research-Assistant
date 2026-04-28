"""
utils/llm.py

Centralized Groq LLM client.
"""

import os
from groq import Groq

_PRIMARY_MODEL = "llama-3.3-70b-versatile"
_client: Groq | None = None

def get_client() -> Groq:
    """Return a cached Groq client."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        _client = Groq(api_key=api_key)
        print("[INFO] Groq client initialized.")
    return _client

def generate_text(prompt: str) -> str:
    """Send a prompt to Groq and return the response."""
    client = get_client()
    messages = [{"role": "user", "content": prompt}]

    try:
        print(f"[INFO] Groq call -> {_PRIMARY_MODEL}")
        response = client.chat.completions.create(
            model=_PRIMARY_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[ERROR] LLM call failed: {exc}")
        return f"[ERROR] LLM call failed: {exc}"
