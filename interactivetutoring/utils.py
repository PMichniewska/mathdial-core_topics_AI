import json
import re
import time
import openai
from typing import List, Dict

from config import get_client

MAX_RETRIES = 5
# Errors that retrying cannot fix (bad key, unknown model, unsupported parameter)
NON_RETRYABLE_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError,
                        openai.NotFoundError, openai.UnprocessableEntityError)

# Dialogue-act tag at the start of a MathDial utterance, e.g. "Teacher: (focus)Okay"
ACT_TAG_RE = re.compile(r"^(Teacher|Student):\s*\([^)]*\)", re.MULTILINE)


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def strip_act_tags(text: str) -> str:
    """Remove dialogue-act tags from every utterance in text: "Teacher: (focus)Okay" -> "Teacher: Okay"."""
    return ACT_TAG_RE.sub(r"\1: ", text)


def call_with_retries(**kwargs):
    """Call the chat completions API with kwargs and return the response.

    Temporary errors (timeouts, rate limits, server errors) are retried up to MAX_RETRIES
    times with exponential backoff; NON_RETRYABLE_ERRORS are raised immediately."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return get_client().chat.completions.create(**kwargs)
        except NON_RETRYABLE_ERRORS:
            raise
        except openai.OpenAIError as e:
            print(f"API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
