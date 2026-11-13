from typing import Dict


class Message(object):
    def __init__(self, persona: str, text: str):
        self.persona = persona
        self.text = text.replace('\n', '\\n')

    def __str__(self):
        # .value: on Python 3.11+ f"{Roles.STUDENT}" gives "Roles.STUDENT" instead of "Student"
        return f"{self.persona.value}: {self.text}"

    def to_gpt_format(self, role_mapping: Dict[str, str]):
        return {"role": role_mapping.get(self.persona, "unknown"), "content": self.text}
