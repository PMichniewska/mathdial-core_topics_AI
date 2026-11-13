import os
from pathlib import Path

import openai
from dotenv import load_dotenv

# Credentials are read from the git-ignored .env file in the repo root (see .env.example)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.tilburg.ai/v1")

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gpt-6-luna")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "gpt-4.1-nano")
