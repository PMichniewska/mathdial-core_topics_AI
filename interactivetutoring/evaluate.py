from re import search
from typing import Dict, List, Tuple
import argparse

from roles import Roles
from utils import read_jsonl, strip_act_tags


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../output/chatgpt_baseline.jsonl")
    parser.add_argument("--model_name", type=str, default="chatgpt_baseline")
    parser.add_argument("--max_k", type=int, default=7)
    parser.add_argument("--ground_truth", action="store_true",
                        help="Evaluate the human dialogues in the 'conversation' field (e.g. data/test.jsonl) "
                             "instead of generated ones; --model_name is ignored")
    return parser.parse_args()


def findall(regexp: str, text: str):
    matches = []
    while True:
        res = search(regexp, text)
        if res:
            matches.append(text[res.start():res.end()])
            text = text[res.end():]
        else:
            break
    return matches


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_match(text: str, gt_answer: str):
    all_numbers = findall(r'[0-9]+(\.[0-9]+)?', text.replace(",", ""))
    answer = gt_answer.replace(",", "")
    if _is_float(answer):
        float_answer = float(answer)
        for number in all_numbers:
            if _is_float(number):
                float_number = float(number)
                if abs(float_answer - float_number) < 1e-7:
                    return True
    else:
        return answer in all_numbers


def all_match(role: str, conversation: List[str], answer: str) -> float:
    count = 0
    for utterance in conversation:
        utterance = utterance.strip().strip("\n")
        if utterance.startswith(role):
            if get_match(utterance, answer):
                return count
            count += 1.
    return float('inf')


def evaluate_at_k(utterances_list: List[str], answer, k) -> (bool, bool):
    student = all_match(Roles.STUDENT + ":", utterances_list, answer)
    teacher = all_match(Roles.TEACHER + ":", utterances_list, answer)
    is_success_at_k = student <= k
    is_telling_at_k = teacher <= min(student, k)
    return is_success_at_k, is_telling_at_k


def extract_answer(ground_truth: str) -> str:
    """Return the numerical final answer, which is on the last line of a ground-truth solution."""
    ground_truth_solution = ground_truth.split("\n")[-1].strip()
    return ground_truth_solution.split("\\n")[-1].strip()


def human_conversation(problem: Dict) -> str:
    """Convert a human MathDial dialogue to the generated format: <EOM> delimiters, no act tags."""
    return "<EOM>".join(strip_act_tags(utterance.strip()) for utterance in problem["conversation"].split("|EOM|"))


def evaluate_problems(data: List[Dict], model_name: str, max_k: int,
                      ground_truth: bool = False) -> Tuple[List[float], List[float]]:
    """Compute Success@k and Telling@k for k = 0..max_k (index 0 is unused) over all problems.

    The conversation is read from problem[model_name], or from the human dialogue if ground_truth is set."""
    dataset_success_at_k = [0] * int(max_k + 1)
    dataset_telling_at_k = [0] * int(max_k + 1)

    for problem in data:
        conversation = human_conversation(problem) if ground_truth else problem[model_name]
        conversation_utterances_list = conversation.split("<EOM>")
        numerical_solution = extract_answer(problem["ground_truth"])
        for k_point in range(1, max_k + 1):
            success_at_k, telling_at_k = evaluate_at_k(conversation_utterances_list, numerical_solution, k_point)
            dataset_success_at_k[k_point] += success_at_k / len(data)
            dataset_telling_at_k[k_point] += telling_at_k / len(data)
    return dataset_success_at_k, dataset_telling_at_k


if __name__ == '__main__':
    args = get_args()
    data = read_jsonl(args.input_file)
    dataset_success_at_k, dataset_telling_at_k = evaluate_problems(data, args.model_name, args.max_k,
                                                                   args.ground_truth)
    print(f"Success@{args.max_k}:", *dataset_success_at_k)
    print(f"Telling@{args.max_k}:", *dataset_telling_at_k)
