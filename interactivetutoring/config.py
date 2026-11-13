import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# OPENAI_API_KEY is read from the environment. A git-ignored .env file in the repo root
# (see .env.example) is used as a fallback; variables already set in the shell take precedence.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = os.getenv("OPENAI_API_BASE", "https://api.tilburg.ai/v1")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gpt-6-luna")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "gpt-4.1-nano")


@lru_cache(maxsize=None)
def get_client() -> OpenAI:
    """Return the shared OpenAI client, created on first use so that scripts which make
    no API calls (e.g. evaluate.py) work without an API key.

    The SDK's own retries are disabled because utils.call_with_retries handles retries."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your shell before running.")
    return OpenAI(api_key=api_key, base_url=API_BASE, max_retries=0, timeout=120)
