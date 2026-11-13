# 🧮 MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems
[ArXiv paper](https://arxiv.org/abs/2305.14536)

[HuggingFace Hub](https://huggingface.co/datasets/eth-nlped/mathdial)

[EMNLP 2023 Video](https://s3.amazonaws.com/pf-user-files-01/u-59356/uploads/2023-11-13/6a13y89/mathdial-emnlp23-v5.mp4)

[![CC BY 4.0][cc-by-shield]][cc-by]

🧮 MathDial is grounded in math word problems as well as student confusions which provide a challenging testbed for creating faithful and equitable dialogue tutoring models able to reason over complex information. 
Current models achieve high accuracy in solving such problems but they fail in the task of teaching.

# Description
>While automatic dialogue tutors hold great potential in making education personalized and
more accessible, research on such systems has been hampered by a lack of sufficiently
large and high-quality datasets. Collecting such datasets remains challenging, as recording
tutoring sessions raises privacy concerns and crowdsourcing leads to insufficient data quality. 
To address this, we propose a framework to generate such dialogues by pairing human teachers 
with a Large Language Model (LLM) prompted to represent common student errors. 
We describe how we use this framework to collect 🧮 MathDial, a dataset of 3k one-to-one
teacher-student tutoring dialogues grounded in multi-step math reasoning problems. While
models like GPT-3 are good problem solvers, they fail at tutoring because they generate factually incorrect feedback or are prone to revealing solutions to students too early. To overcome
this, we let teachers provide learning opportunities to students by guiding them using various
scaffolding questions according to a taxonomy of teacher moves. We demonstrate 🧮 MathDial and its extensive annotations can be used to finetune models to be more effective tutors
(and not just solvers). We confirm this by automatic and human evaluation, notably in an
interactive setting that measures the trade-off between student solving success and telling solutions. 
The dataset is released publicly.


# Dataset
The dataset is available in the data folder. It contains all 2861 conversations. 
The dataset is split into train and test - see `data/train.csv` and `data/test.csv` (for csv format). 
For jsonl format, see `data/train.jsonl` and `data/test.jsonl`. To see a small sample of the dataset, look at `data/example.jsonl`
Please note that each row in the file consists of full conversations between a teacher and a student delimited with special `|EOM|` notation. 

![dataset-comparison](images/image.png)

## Data Structure
- `qid` - unique identifier of the problem
- `scenario` - order of the problem in the data collection, out of the 5 scenarios in a session
- `question` - math problem text
- `ground_truth` - correct answer to the problem
- `student_incorrect_solution` - student incorrect solution to the problem caused by some confusion
- `student_profile` - student profile based on general math problem solving student misconceptions
- `teacher_described_confusion` - teacher annotated student confusion in free text
- `self-correctness` - teacher annotated whether student solved the problem correctly by the end of the conversation
    - options: `Yes`, `Yes, but I had to reveal the answer`, `No`
- `self-typical-confusion` - teacher annotated whether student exhibited a typical 7th grade confusion, Likert scale 1 (unlikely) to 5 (very likely)
- `self-typical-interactions` - teacher annotated whether student exhibited typical 7th grade interactions, Likert scale 1 (unlikely) to 5 (very likely)
- `conversation` - conversation in a string format with `|EOM|` delimiter between Teacher and Student personas  `Persona: (dialog_act) text` e.g. `Teacher: (focus) What is the difference?|EOM|Student: I mean ...|EOM|Teacher:`


## HuggingFace Loader

# Setup your environment
```bash 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your-api-token>"
```

API settings are read from environment variables by `interactivetutoring/config.py`. A git-ignored `.env` file (see `.env.example`) is used as a fallback; variables set in the shell take precedence.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | API token (never commit it) |
| `OPENAI_API_BASE` | `https://api.tilburg.ai/v1` | Any OpenAI-compatible endpoint |
| `TEACHER_MODEL` | `gpt-6-luna` | Teacher model |
| `STUDENT_MODEL` | `gpt-4.1-nano` | Student model |

## Changes in this fork
- Uses the Tilburg University OpenAI-compatible API instead of OpenAI directly; endpoint, key and models are configured via `.env` instead of being hard-coded.
- Teacher model changed from `gpt-3.5-turbo` to `gpt-6-luna` (configurable via `TEACHER_MODEL`).
- Student model changed from `gpt-3.5-turbo-instruct` to `gpt-4.1-nano`, and the student now uses the chat completions endpoint instead of the legacy completions endpoint. Student temperature raised from 0 to 0.4 (gpt-4.1-nano is a non-reasoning model, so no reasoning setting applies).
- API calls no longer retry forever: temporary errors (timeouts, rate limits) are retried up to 5 times, and errors retrying cannot fix (bad key, unknown model, unsupported parameter) stop the script immediately.
- Added `.gitignore` (excludes `.env` and the virtual env), `.env.example`, and `python-dotenv`/`tqdm` to `requirements.txt`.

# Replication with GPT-6 Luna
This fork replicates the interactive tutoring evaluation of Macina et al. (2023) with `gpt-6-luna` as the tutor, comparing 0-, 2- and 5-shot prompting, plus the NextStep baseline to calibrate the new simulated student. Metrics are Success@k (the student states the correct answer within k turns) and Telling@k (the tutor states it first), for k = 1..10.

## What changed and why
- **Models.** The paper's models (`text-davinci-003`, `gpt-3.5-turbo-instruct`, `gpt-3.5-turbo`) are retired. The tutor is `gpt-6-luna` and the simulated student is `gpt-4.1-nano`, both through the Tilburg University OpenAI-compatible API, using the current `openai` SDK.
- **Tutor prompt.** `TEACHER_BASE` in `teachers.py` is the tutor prompt from the paper's Appendix E. Tutor calls use `reasoning_effort="none"` (`--tutor_effort`), `max_completion_tokens=512` and `stop=["Student:", "student:"]`. Before each run, one test request checks these parameters and drops any the API rejects, with a warning.
- **Student.** `STUDENT_PROMPT` is unchanged and identical across all conditions. The student runs at temperature 0.4 instead of 0. `STUDENT_PROMPT` was written for completion models; sent as a chat message, `gpt-4.1-nano` answered it as an assistant and started tutoring the student after its first turn. A one-sentence system message (`STUDENT_SYSTEM_PROMPT` in `students.py`) therefore tells the model to write only the student's next message, in the first person, and never speak as the teacher.
- **Few-shot tutor.** `FewShotTeacher` puts n example dialogues from `data/train.jsonl` at the start of the system prompt, under "Here are examples of good tutoring conversations:", followed by the Appendix E prompt. Putting the examples first means the prompt starts with the same text for every problem, so prompt caching can reuse it across problems. A pool of 10 dialogues (distinct qids) is sampled once with a fixed seed, from training dialogues that show good tutoring only: the teacher reported that the student solved the problem without being told the answer (`self-correctness == "Yes"`), and by the evaluator's criteria the student states the correct answer before the teacher does (1,474 of 2,262 training dialogues qualify). Problems that appear anywhere in `data/test.jsonl` (same qid or same question text) are excluded from the pool: 585 of 2,262 training dialogues share a problem with the test set, so without this the tutor could see a test problem's own dialogue. 1,090 training dialogues are both good and test-disjoint. Every test problem gets the same first n pool dialogues. Dialogue-act tags (e.g. `(focus)`) are stripped from the examples. 0-shot is exactly the Appendix E prompt.
- **Early stopping.** A conversation ends as soon as a student message contains the correct final answer (same answer extraction and matching as `evaluate.py`), or after 10 student turns. Both metrics only depend on the first time the answer appears, so this does not change the scores and saves calls.
- **Runner.** `gpt_baseline.py` runs conversations in parallel, appends each finished conversation to the output file immediately, and skips already finished items on restart (item id = line index in the input file, because qids are not unique in `test.jsonl`). Token usage of every call is logged to `output/{run}_usage.csv`, and each run is summarised in `output/run_log.csv`.
- **Bug fix.** On Python 3.11+ `f"{Roles.STUDENT}"` gives `Roles.STUDENT` instead of `Student`, so generated conversations never matched the evaluator's `Student:` prefix (every score was 0) and the student saw `Roles.TEACHER:` in its dialogue history. `Message.__str__` now uses the role's value.
- **Evaluator.** `evaluate.py --ground_truth` evaluates the human dialogues in `data/test.jsonl` (act tags stripped, `|EOM|` replaced by `<EOM>`).

## Running
```bash
source .venv/bin/activate
export OPENAI_API_KEY="<your-api-token>"

# Pilot on the first N test items (add --verbose to print each conversation)
python interactivetutoring/gpt_baseline.py --teacher nextstep --limit 20 --output_dir output/pilot
python interactivetutoring/gpt_baseline.py --teacher fewshot --shots 0 --limit 20 --output_dir output/pilot

# Full test set (599 dialogues) for every condition
python interactivetutoring/gpt_baseline.py --teacher nextstep
python interactivetutoring/gpt_baseline.py --teacher fewshot --shots 0
python interactivetutoring/gpt_baseline.py --teacher fewshot --shots 2
python interactivetutoring/gpt_baseline.py --teacher fewshot --shots 5

# Evaluate one condition, or the human dialogues
python interactivetutoring/evaluate.py --input_file output/fewshot_2shot.jsonl --model_name fewshot_2shot --max_k 10
python interactivetutoring/evaluate.py --input_file data/test.jsonl --ground_truth --max_k 10

# Results table (results/results_table.csv, .md) and curves (results/curves.png)
python scripts/make_results.py
```
Other flags: `--workers` (default 16), `--seed` (few-shot pool, default 42), `--max_utterances` (default 10), `--tutor_effort` (default `none`). Output goes to `output/{teacher}_{shots}shot.jsonl`, with the conversation stored under `--model_name` (default `{teacher}_{shots}shot`).

## Results
Full test set (599 dialogues), tutor `gpt-6-luna` (`reasoning_effort="none"`), student `gpt-4.1-nano`. Produced by `scripts/make_results.py` (`results/results_table.md`, `results/curves.png`).

| Condition | n | Success@5 | Telling@5 | Success@10 | Telling@10 |
|---|---:|---:|---:|---:|---:|
| NextStep (ours) | 599 | 61.1% | 0.0% | 72.8% | 0.0% |
| GPT-6 Luna 0-shot | 599 | 99.3% | 3.7% | 99.5% | 4.0% |
| GPT-6 Luna 2-shot | 599 | 98.8% | 2.3% | 98.8% | 2.3% |
| GPT-6 Luna 5-shot | 599 | 99.3% | 2.2% | 99.3% | 2.2% |
| Ground truth (repo evaluator) | 599 | 72.0% | 8.2% | 86.1% | 12.2% |
| NextStep (paper) |  | 25% | 0% | 25% | 0% |
| ChatGPT (paper) |  | 29% | 16% | 53% | 32% |
| Flan-T5 780M (paper) |  | 30% | <1% | 39% | 4% |
| Ground truth (paper) |  | 59% | <1% | 82% | 4% |

![Success@k and Telling@k](results/curves.png)

- **Ceiling effect.** All three GPT-6 Luna conditions reach about 99% Success@5, so 0-, 2- and 5-shot prompting cannot be told apart on success. They differ only at k = 1 (0-shot 71.8%, 2-shot 64.8%, 5-shot 67.8%) and slightly on telling: few-shot examples roughly halve Telling@10 (4.0% for 0-shot, 2.3% and 2.2% with examples).
- **The student is much easier to convince than the paper's.** With NextStep, which gives no hints, the student reaches the answer in 72.8% of dialogues by k = 10, against 25% in the paper. In 161 dialogues (26.9%) the answer already appears in the student's first turn, close to the paper's flat 25%. This is mostly a matching artifact: in 150 of 599 test items the student's incorrect solution already contains the correct number, for example as an intermediate result, so restating it counts as success. The remaining 46 points come from `gpt-4.1-nano` correcting itself without help over later turns, despite the persona line "Only when the teacher provides several good reasoning questions, Kayla understands the problem and corrects the solution."
- **Compared with the paper.** GPT-6 Luna tells far less than the paper's ChatGPT (2–4% vs. 32% Telling@10), but its high success rate mostly reflects the easier student: even NextStep beats every tutor in the paper.

## Cost and runtime
Cost at $0.10 per 1M input and $0.50 per 1M output tokens (from `output/run_log.csv`), 16 workers. Cached input tokens are charged at the full rate here, so these are upper bounds.

| Condition | API calls | Input tokens (cached) | Output tokens | Cost | Runtime |
|---|---:|---:|---:|---:|---:|
| NextStep | 3,163 | 2,326,254 (311,168) | 348,768 | $0.41 | 130 s |
| 0-shot | 1,910 | 884,174 (22,844) | 181,916 | $0.18 | 81 s |
| 2-shot | 2,048 | 2,186,827 (806,953) | 201,448 | $0.32 | 118 s |
| 5-shot | 2,019 | 3,605,395 (1,455,061) | 193,289 | $0.46 | 109 s |
| **Full runs** | 9,140 | 9,002,650 | 925,421 | **$1.36** | **7.3 min** |

Smoke tests (3 items) and pilots (20 items) added $0.11, so the whole replication cost about $1.47. NextStep makes the most calls because its student rarely finishes early. Placing the few-shot examples before the problem lets 37–40% of the 2- and 5-shot input tokens come from the prompt cache.

## Limitations
- **Different student.** The simulated student is `gpt-4.1-nano` at temperature 0.4, not the paper's InstructGPT student, so absolute numbers are not directly comparable with the paper. The NextStep baseline shows that this student reaches the answer on its own far more often (72.8% vs. 25% Success@10), which pushes all tutor conditions to the ceiling. The student was deliberately not recalibrated, so that `STUDENT_PROMPT` stays as in the original repo.
- **Cached responses.** The Tilburg API appears to return cached responses for identical requests (a repeated run finished instantly with identical tokens), so rerunning a condition reproduces the same conversations even at temperature 0.4.
- **Evaluator does not reproduce the paper's ground truth.** On the human test dialogues this repo's evaluator gives Success@5/10 = 72%/86% and Telling@5/10 = 8%/12%, versus 59%/82% and <1%/4% reported in the paper. A likely cause is that the numeric matching counts any occurrence of the answer number (e.g. in an intermediate step or a restated question), but the exact difference to the paper's evaluation is unknown. Compare conditions with each other rather than with the paper's absolute numbers.

# Teacher model - Generate next tutor response
```bash
python interactivetutoring/gpt_baseline.py --input_file "<dataset-path>" --model_name "<name-of-your-model>" --max_utterances "<stopping-point>" --export_file "<export-file-with-conversations>"
```
Example using ChatGPT3.5 as a teacher model on a small data sample:
```bash
python interactivetutoring/gpt_baseline.py --input_file "data/example.jsonl" --model_name "chatgpt_baseline" --max_utterances 5 --export_file "output/chatgpt_baseline.jsonl"
```
Please see `interactivetutoring/teachers.py` how to add your own teacher model.

# Evaluate your teacher model using simulated student
```bash
python interactivetutoring/evaluate.py --input_file "<path-to-file-with-generations>" --model_name "<model-name>"  
```
Example using dummy data:
```bash
python interactivetutoring/evaluate.py --input_file "output/example_model_output.jsonl" --model_name "chatgpt_baseline"  
```

## Supervised Fine-Tuning (SFT)
To train the model using SFT, navigate to the `SFT_Finetuning` folder and run the script. Or run the following command:
```bash 
cd SFT_Finetuning
python SFT_Finetuning.py
```
This will fine-tune the Qwen/Qwen2.5-1.5B-Instruct model on the MathDial dataset. The model is available at [Huggingface eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct](https://huggingface.co/eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct).

# Citation
Please cite the following:
> Jakub Macina, Nico Daheim, Sankalan Chowdhury, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023. MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5602–5621, Singapore. Association for Computational Linguistics.

# Example
![overview](images/generations.png)

# License
The MathDial dataset is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].
If you use it, please cite our paper (see [Citation](#citation)).

The math word problems originate from [GSM8K](https://github.com/openai/grade-school-math) (MIT License).

_Earlier versions of this README listed CC BY-SA 4.0. The license was aligned with the Hugging Face dataset card (CC BY 4.0) in November 2026._

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
