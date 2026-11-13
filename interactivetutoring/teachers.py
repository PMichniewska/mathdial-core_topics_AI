import random
from functools import lru_cache
from typing import Dict, List, Tuple

import openai

from config import TEACHER_MODEL
from evaluate import evaluate_at_k, extract_answer, human_conversation
from history import History
from roles import Roles
from utils import call_with_retries, read_jsonl, strip_act_tags

# Tutor prompt from Macina et al. (2023), Appendix E
TEACHER_BASE = """A tutor and a student work together to solve the following math word problem.
Math problem: {problem}
The correct solution is as follows:
{ground_truth}
Your role is tutor. The tutor is a soft-spoken empathetic person who dislikes giving out direct answers to students and instead likes to answer with other questions that would help the student understand the concepts so students can solve the problem themselves."""

EXAMPLES_HEADER = "Here are examples of good tutoring conversations:"
EXAMPLE_POOL_SIZE = 10

# Default tutor call parameters; check_tutor_params() removes any the API rejects
TUTOR_PARAMS = {
    "reasoning_effort": "none",
    "max_completion_tokens": 512,
    "stop": ["Student:", "student:"],
}


def check_tutor_params(params: Dict) -> Tuple[Dict, List[str]]:
    """Send one short test request with params and drop each parameter the API rejects.

    Returns the accepted parameters and the names of the dropped ones. Errors that do not
    name one of the parameters (e.g. a bad key or unknown model) are raised."""
    params = dict(params)
    dropped = []
    while True:
        try:
            call_with_retries(model=TEACHER_MODEL, messages=[{"role": "user", "content": "Say hi."}], **params)
            return params, dropped
        except openai.BadRequestError as e:
            rejected = e.param if e.param in params else next((p for p in params if p in str(e)), None)
            if rejected is None:
                raise
            print(f"WARNING: {TEACHER_MODEL} rejected parameter '{rejected}' ({e}); dropping it.")
            params.pop(rejected)
            dropped.append(rejected)


def is_good_example(example: Dict) -> bool:
    """Return True if a training dialogue shows successful tutoring without telling.

    The teacher must report that the student solved the problem without the answer being revealed
    ("self-correctness" == "Yes"), and by the evaluator's own criteria the student must state the correct
    answer and the teacher must not state it first."""
    if example["self-correctness"] != "Yes":
        return False
    success, telling = evaluate_at_k(human_conversation(example).split("<EOM>"),
                                     extract_answer(example["ground_truth"]), float("inf"))
    return success and not telling


@lru_cache(maxsize=None)
def load_example_pool(train_path: str, test_path: str, seed: int) -> Tuple[Dict, ...]:
    """Sample EXAMPLE_POOL_SIZE good training dialogues (see is_good_example) with distinct qids,
    reproducibly for a given seed.

    Dialogues whose problem appears anywhere in test_path (same qid or same question text) are never
    sampled, so no problem shown as an example is ever used for testing."""
    test = read_jsonl(test_path)
    test_qids = {example["qid"] for example in test}
    test_questions = {example["question"].strip() for example in test}
    train = read_jsonl(train_path)
    random.Random(seed).shuffle(train)
    pool, seen_qids = [], set()
    for example in train:
        if example["qid"] in test_qids or example["question"].strip() in test_questions:
            continue
        if example["qid"] not in seen_qids and is_good_example(example):
            pool.append(example)
            seen_qids.add(example["qid"])
        if len(pool) == EXAMPLE_POOL_SIZE:
            break
    return tuple(pool)


def format_example(example: Dict) -> str:
    """Format a training dialogue as problem, correct solution and act-tag-free conversation."""
    conversation = strip_act_tags(example["conversation"].replace("|EOM|", "\n"))
    return (f"Math problem: {example['question']}\n"
            f"The correct solution is as follows:\n{example['ground_truth']}\n"
            f"Conversation:\n{conversation}")


class NextStepTeacher(object):
    """Baseline tutor that makes no API call and always asks for the next step."""

    def __init__(self):
        self.persona = Roles.TEACHER
        self.name = "NextStep"
        self.last_usage = None

    def reset(self):
        pass

    def response(self, history: History, question: str, ground_truth_solution: str, qid: str = None):
        """Return the fixed NextStep utterance."""
        return "What is the next step?"


class FewShotTeacher(object):
    """Tutor that prompts TEACHER_MODEL with n_shots example dialogues followed by the Appendix E prompt.

    The examples are the first n_shots dialogues of a fixed pool (see load_example_pool) that contains no
    problem from the test set, so every test problem gets the same examples. They come before the
    problem-specific Appendix E prompt so the system prompt starts with the same text for every problem,
    which lets prompt caching reuse it across problems. n_shots=0 gives exactly the Appendix E prompt."""

    def __init__(self, n_shots: int, train_path: str, test_path: str, seed: int = 42, params: Dict = None):
        self.persona = Roles.TEACHER
        self.name = f"{TEACHER_MODEL} ({n_shots}-shot)"
        self.n_shots = n_shots
        self.pool = load_example_pool(train_path, test_path, seed) if n_shots > 0 else ()
        self.params = TUTOR_PARAMS if params is None else params
        self.last_usage = None  # token usage of the most recent API call

    def reset(self):
        pass

    def system_prompt(self, question: str, ground_truth_solution: str, qid: str = None) -> str:
        """Build the system prompt for one test problem."""
        prompt = TEACHER_BASE.format(problem=question, ground_truth=ground_truth_solution)
        if self.n_shots == 0:
            return prompt
        examples = self.pool[:self.n_shots]
        formatted = "\n\n".join(f"Example {i}:\n{format_example(example)}" for i, example in enumerate(examples, 1))
        return f"{EXAMPLES_HEADER}\n\n{formatted}\n\n{prompt}"

    def response(self, history: History, question: str, ground_truth_solution: str, qid: str = None):
        """Generate the next tutor utterance for the conversation so far."""
        messages = [{"role": "system", "content": self.system_prompt(question, ground_truth_solution, qid)},
                    *history.to_gpt_messages({Roles.TEACHER: "assistant", Roles.STUDENT: "user"})]
        response = call_with_retries(model=TEACHER_MODEL, messages=messages, **self.params)
        self.last_usage = response.usage
        return (response.choices[0].message.content or "").strip()
