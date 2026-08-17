from ..settings import Persona
from ..settings import Settings


def build_system_prompt(persona: Persona, fillers: bool = True) -> str:
    filler_line = ""
    if fillers and persona.filler_words:
        words = ", ".join(f"'{w}...'" for w in persona.filler_words)
        filler_line = (
            f"Always open your reply with a short spoken filler word "
            f"such as {words}. Start speaking with the filler within the "
            f"first 2 words of your response, then give the real answer. "
            f"Never open with a filler if it sounds forced.\n"
        )
    return (
        f"You are {persona.name}.\n"
        f"{persona.role}\n"
        f"Language: answer in {persona.language}. Use words a speaking "
        f"person would use; prefer short sentences of 4-9 words.\n"
        f"{filler_line}"
        f"Instructions: no markdown, no bullet lists, no URLs, no "
        f"punctuation beyond commas and full stops, reply in Roman script, "
        f"and never repeat the caller's request back to them."
    )


class TurnMemory:
    def __init__(self, settings: Settings):
        self._system = build_system_prompt(
            settings.persona, fillers=settings.llm.fillers
        )
        self._max_turns = settings.call.max_turns
        self._messages: list[dict] = [{"role": "system", "content": self._system}]

    def add_user(self, text: str) -> list[dict]:
        self._messages.append({"role": "user", "content": text})
        return self.snapshot()

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def snapshot(self) -> list[dict]:
        if len(self._messages) <= 2 + self._max_turns * 2:
            return list(self._messages)
        head = self._messages[:1]
        tail = self._messages[-(self._max_turns * 2) :]
        self._messages = head + tail
        return list(self._messages)

    def reset(self) -> None:
        self._messages = self._messages[:1]