"""
utils/config.py

Loads environment variables from .env at project root.
Uses override=True to ensure stale values are always refreshed in
long-running processes like Streamlit.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not found in environment.")
    print(f"[DEBUG] Loaded GROQ_API_KEY: {key[:5]}***")
    return key


def get_tavily_api_key() -> str:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise ValueError("TAVILY_API_KEY not found in environment.")
    return key
