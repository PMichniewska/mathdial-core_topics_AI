import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Optional, Set

from tqdm import tqdm as tqdm

from config import STUDENT_MODEL, TEACHER_MODEL
from evaluate import extract_answer, get_match
from history import History
from message import Message
from roles import Roles
from students import STUDENT_SYSTEM_PROMPTS, InstructGPTStudent
from teachers import TUTOR_PARAMS, FewShotTeacher, NextStepTeacher, check_tutor_params
from utils import read_jsonl

PRICE_PER_M_INPUT = 0.10
PRICE_PER_M_OUTPUT = 0.50
TOKEN_FIELDS = ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"]
USAGE_FIELDS = ["run_id", "item_id", "turn", "role", "model", *TOKEN_FIELDS]
RUN_LOG_FIELDS = ["run_id", "run_name", "new_conversations", "total_conversations", "seconds", "workers",
                  "tutor_effort", "dropped_params", "calls", *TOKEN_FIELDS, "cost_usd"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/test.jsonl")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--teacher", type=str, choices=["nextstep", "fewshot"], default="fewshot")
    parser.add_argument("--shots", type=int, choices=[0, 2, 5], default=0)
    parser.add_argument("--tutor_effort", type=str, default="none", help="reasoning_effort of the tutor model")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N test items (pilots)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling the few-shot example pool")
    parser.add_argument("--max_utterances", type=int, default=10)
    parser.add_argument("--student", type=str, choices=list(STUDENT_SYSTEM_PROMPTS), default="original",
                        help="student variant (see students.py)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="default: output for the original student, output/{student} otherwise")
    parser.add_argument("--export_file", type=str, default=None,
                        help="default: {output_dir}/{teacher}_{shots}shot.jsonl")
    parser.add_argument("--model_name", type=str, default=None,
                        help="key under which the conversation is stored; default: {teacher}_{shots}shot")
    parser.add_argument("--verbose", action="store_true", help="print every finished conversation")
    return parser.parse_args()


def print_conversation(question: str, ground_truth_solution: str, incorrect_solution: str, history: History):
    print("\n\n## Conversation")
    print(f"Question: {question}")
    print(f"Correct solution: {ground_truth_solution}")
    print(f"Incorrect solution: {incorrect_solution}")
    print(history)


def usage_tokens(usage) -> Optional[Dict[str, int]]:
    """Extract input, cached, output and reasoning token counts from a response.usage object.

    Reasoning tokens are part of output_tokens, and cached tokens are part of input_tokens."""
    if usage is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {"input_tokens": usage.prompt_tokens or 0,
            "cached_tokens": getattr(prompt_details, "cached_tokens", None) or 0,
            "output_tokens": usage.completion_tokens or 0,
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None) or 0}


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Cost at PRICE_PER_M_INPUT / PRICE_PER_M_OUTPUT dollars per million tokens."""
    return input_tokens / 1e6 * PRICE_PER_M_INPUT + output_tokens / 1e6 * PRICE_PER_M_OUTPUT


def load_done_item_ids(export_file: str) -> Set[int]:
    """Return the item ids already in export_file, dropping any incomplete last line left by a crash."""
    if not os.path.exists(export_file):
        return set()
    with open(export_file, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"WARNING: dropping unreadable line in {export_file}")
    if len(records) != len(lines):
        with open(export_file, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(record) + "\n" for record in records)
    return {record["item_id"] for record in records}


class RunWriter(object):
    """Thread-safe writer that appends finished conversations to the export file and the token usage
    of every API call to the usage CSV, keeping running token totals for this run."""

    def __init__(self, export_file: str, usage_file: str, run_id: str):
        self.lock = threading.Lock()
        self.run_id = run_id
        self.export_fh = open(export_file, "a", encoding="utf-8")
        new_usage_file = not os.path.exists(usage_file) or os.path.getsize(usage_file) == 0
        self.usage_fh = open(usage_file, "a", newline="")
        self.usage_writer = csv.DictWriter(self.usage_fh, USAGE_FIELDS)
        if new_usage_file:
            self.usage_writer.writeheader()
        self.calls = 0
        self.totals = dict.fromkeys(TOKEN_FIELDS, 0)

    def log_usage(self, item_id: int, turn: int, role: str, model: str, usage):
        """Record the token usage of one API call (no-op for usage=None, e.g. the NextStep tutor)."""
        tokens = usage_tokens(usage)
        if tokens is None:
            return
        with self.lock:
            self.usage_writer.writerow({"run_id": self.run_id, "item_id": item_id, "turn": turn, "role": role,
                                        "model": model, **tokens})
            self.usage_fh.flush()
            self.calls += 1
            for field in TOKEN_FIELDS:
                self.totals[field] += tokens[field]

    def write_conversation(self, record: Dict):
        """Append one finished conversation to the export file."""
        with self.lock:
            self.export_fh.write(json.dumps(record) + "\n")
            self.export_fh.flush()

    def close(self):
        self.export_fh.close()
        self.usage_fh.close()


def run_conversation(item_id: int, problem: Dict, teacher, model_name: str, max_utterances: int,
                     writer: RunWriter, student_variant: str = "original") -> History:
    """Simulate one tutoring conversation and write it to the export file.

    The conversation ends as soon as a student message contains the correct final answer (Success@k and
    Telling@k only depend on its first occurrence), or after max_utterances student messages."""
    student = InstructGPTStudent(student_variant)
    question = problem["question"]
    ground_truth_solution = problem["ground_truth"]
    incorrect_solution = problem["student_incorrect_solution"]
    answer = extract_answer(ground_truth_solution)

    history = History()
    history.add_message(Message(Roles.TEACHER, "Hi " + student.name + "! Could you walk me through your solution?"))

    solved = False
    for turn in range(max_utterances):
        student_message = Message(Roles.STUDENT, student.response(history, question, incorrect_solution))
        history.add_message(student_message)
        writer.log_usage(item_id, turn, "student", STUDENT_MODEL, student.last_usage)
        if get_match(str(student_message), answer):
            solved = True
            break

        teacher_response_message = Message(Roles.TEACHER, teacher.response(history, question, ground_truth_solution,
                                                                           qid=problem["qid"]))
        history.add_message(teacher_response_message)
        writer.log_usage(item_id, turn, "teacher", TEACHER_MODEL, teacher.last_usage)

    record = {**problem, "item_id": item_id, "model_name": model_name, "solved": solved}
    record[model_name] = history.to_delimited_string("<EOM>")
    writer.write_conversation(record)
    return history


if __name__ == '__main__':
    args = get_args()
    if args.teacher == "nextstep":
        args.shots = 0
    if args.output_dir is None:
        args.output_dir = "output" if args.student == "original" else os.path.join("output", args.student)
    run_name = f"{args.teacher}_{args.shots}shot"
    model_name = args.model_name or run_name
    export_file = args.export_file or os.path.join(args.output_dir, f"{run_name}.jsonl")
    usage_file = os.path.splitext(export_file)[0] + "_usage.csv"
    os.makedirs(os.path.dirname(export_file) or ".", exist_ok=True)

    data = read_jsonl(args.input_file)[:args.limit]
    done = load_done_item_ids(export_file)
    todo = [(item_id, problem) for item_id, problem in enumerate(data) if item_id not in done]
    print(f"{run_name}: {len(data)} items, {len(data) - len(todo)} already in {export_file}, {len(todo)} to run")

    dropped = []
    if args.teacher == "fewshot" and todo:
        tutor_params, dropped = check_tutor_params({**TUTOR_PARAMS, "reasoning_effort": args.tutor_effort})
    make_teacher = (NextStepTeacher if args.teacher == "nextstep"
                    else lambda: FewShotTeacher(args.shots, args.train_file, args.input_file, args.seed,
                                                  tutor_params))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = RunWriter(export_file, usage_file, run_id)
    start = time.time()
    finished = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_conversation, item_id, problem, make_teacher(), model_name,
                                       args.max_utterances, writer, args.student): problem
                       for item_id, problem in todo}
            try:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    history = future.result()
                    finished += 1
                    if args.verbose:
                        problem = futures[future]
                        print_conversation(problem["question"], problem["ground_truth"],
                                           problem["student_incorrect_solution"], history)
            except BaseException:
                # On an API error or Ctrl+C, cancel queued conversations; running ones still finish and are saved
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        writer.close()
        seconds = time.time() - start
        totals = writer.totals
        cost = cost_usd(totals["input_tokens"], totals["output_tokens"])
        print(f"\n{run_name}: {finished} new conversations in {seconds:.0f}s "
              f"({len(done) + finished}/{len(data)} done) -> {export_file}")
        if dropped:
            print(f"Tutor parameters rejected by the API and dropped: {', '.join(dropped)}")
        print(f"Tokens: input {totals['input_tokens']:,} (cached {totals['cached_tokens']:,}), "
              f"output {totals['output_tokens']:,} (reasoning {totals['reasoning_tokens']:,}) "
              f"over {writer.calls} calls")
        print(f"Cost: ${cost:.4f} (${PRICE_PER_M_INPUT}/1M input, ${PRICE_PER_M_OUTPUT}/1M output)")

        run_log = os.path.join(os.path.dirname(export_file) or ".", "run_log.csv")
        new_run_log = not os.path.exists(run_log)
        with open(run_log, "a", newline="") as fh:
            log_writer = csv.DictWriter(fh, RUN_LOG_FIELDS)
            if new_run_log:
                log_writer.writeheader()
            log_writer.writerow({"run_id": run_id, "run_name": run_name, "new_conversations": finished,
                                 "total_conversations": len(done) + finished, "seconds": round(seconds, 1),
                                 "workers": args.workers, "tutor_effort": args.tutor_effort,
                                 "dropped_params": " ".join(dropped), "calls": writer.calls, **totals,
                                 "cost_usd": round(cost, 6)})
