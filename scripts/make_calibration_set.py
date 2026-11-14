"""Build data/calibration.jsonl: training dialogues for calibrating the simulated student.

The set contains no problem from the test set (same qid or same question text) and none of the few-shot
example problems, has one dialogue per qid, and is sampled reproducibly with --seed.

Usage (from the repo root): python scripts/make_calibration_set.py [--size 50] [--seed 0]"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interactivetutoring"))
from teachers import load_example_pool  # noqa: E402
from utils import read_jsonl  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--output_file", type=str, default="data/calibration.jsonl")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool_seed", type=int, default=42, help="seed of the few-shot example pool to exclude")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test = read_jsonl(args.test_file)
    excluded_qids = {example["qid"] for example in test}
    excluded_qids |= {example["qid"] for example in load_example_pool(args.train_file, args.test_file, args.pool_seed)}
    test_questions = {example["question"].strip() for example in test}

    train = read_jsonl(args.train_file)
    random.Random(args.seed).shuffle(train)
    selected, seen_qids = [], set()
    for example in train:
        if example["qid"] in excluded_qids | seen_qids or example["question"].strip() in test_questions:
            continue
        selected.append(example)
        seen_qids.add(example["qid"])
        if len(selected) == args.size:
            break

    with open(args.output_file, "w") as fh:
        for example in selected:
            fh.write(json.dumps(example) + "\n")
    print(f"Wrote {len(selected)} dialogues to {args.output_file}")
