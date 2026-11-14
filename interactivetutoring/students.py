from config import STUDENT_MODEL
from history import History
from roles import Roles
from utils import call_with_retries

STUDENT_PROMPT = """
Student Persona: (STUDENT PERSONA)

Math problem: (MATH PROBLEM)

Student solution: (STUDENT SOLUTION)

Context: (STUDENT NAME) thinks their answer is correct. Only when the teacher provides several good reasoning questions, (STUDENT NAME) understands the problem and corrects the solution. (STUDENT NAME) can use calculator and thus makes no calculation errors. Send <EOM> tag at end of the student message.

(DIALOG HISTORY)

Student:
"""

STUDENT_NAME = "Kayla"
STUDENT_PERSONA = STUDENT_NAME + " is a 7th grade student. She has problem with understanding of what steps or procedures are required to solve a problem."

# STUDENT_PROMPT was written for completion models (text-davinci-003). Sent as a chat message, chat models
# answer it as an assistant and start tutoring the student, so this system message fixes the role.
STUDENT_SYSTEM_PROMPT = (f"You are role-playing {STUDENT_NAME}, the student in the dialog below. Write only "
                         f"{STUDENT_NAME}'s next message, in the first person. Never speak as the teacher and "
                         f"never address {STUDENT_NAME}.")

# Optional addition for the "calibrated" student. With the original prompt, gpt-4.1-nano corrects its own
# solution even without guidance (NextStep Success@10 = 73% vs. 25% in the paper), so every tutor reaches
# the ceiling. Calibrated against NextStep on data/calibration.jsonl (training problems not in the test set).
STUDENT_CALIBRATION = (f" {STUDENT_NAME} believes her solution is complete and correct. When asked to continue, she "
                       f"says she has already finished and repeats her answer. She only reconsiders a step when the "
                       f"teacher asks a specific question about that step, and then changes only that step.")

STUDENT_SYSTEM_PROMPTS = {
    "original": STUDENT_SYSTEM_PROMPT,
    "calibrated": STUDENT_SYSTEM_PROMPT + STUDENT_CALIBRATION,
}


class InstructGPTStudent(object):
    def __init__(self, variant: str = "original"):
        self.persona = Roles.STUDENT
        self.system_prompt = STUDENT_SYSTEM_PROMPTS[variant]
        self.name = STUDENT_NAME
        self.last_usage = None  # token usage of the most recent API call

    def reset(self):
        pass

    def response(self, history: History, question: str, incorrect_solution: str):
        messages = history.to_delimited_string("<EOM>\n\n")
        prompt = STUDENT_PROMPT.replace("(STUDENT PERSONA)", STUDENT_PERSONA).replace("(STUDENT SOLUTION)",
                                                                                      incorrect_solution).replace(
            "(MATH PROBLEM)", question).replace("(STUDENT NAME)", STUDENT_NAME).replace(
            "(DIALOG HISTORY)", messages)
        response = call_with_retries(
            model=STUDENT_MODEL,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=512,
            stop=["Teacher:", "teacher:"],
        )
        self.last_usage = response.usage
        response = (response.choices[0].message.content or "").strip()
        utterance = response.replace("Student:", "").replace(STUDENT_NAME + ":", "").replace("<EOM>", "").strip(
            "\n")
        return utterance
