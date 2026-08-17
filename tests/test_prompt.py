from vaani.llm.prompt import TurnMemory, build_system_prompt
from vaani.settings import Persona, Settings


def test_system_prompt_has_fillers_and_persona():
    persona = Persona(name="Vaani", role="Be helpful.", filler_words=["Hmm", "Ji"])
    prompt = build_system_prompt(persona, fillers=True)
    assert "Vaani" in prompt
    assert "Hmm" in prompt
    assert "Ji" in prompt
    assert "filler" in prompt


def test_fillers_can_be_disabled():
    persona = Persona(filler_words=["Hmm"])
    prompt = build_system_prompt(persona, fillers=False)
    assert "filler" not in prompt


def test_memory_roundtrip():
    mem = TurnMemory(Settings())
    messages = mem.add_user("Namaste")
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Namaste"}
    mem.add_assistant("Ji, namaste")
    snap = mem.snapshot()
    assert snap[-1] == {"role": "assistant", "content": "Ji, namaste"}


def test_memory_trims_old_turns():
    mem = TurnMemory(Settings())
    for i in range(30):
        mem.add_user(f"q{i}")
        mem.add_assistant(f"a{i}")
    snap = mem.snapshot()
    assert snap[0]["role"] == "system"
    assert len(snap) <= 1 + 12 * 2 + 2


def test_memory_reset_keeps_system():
    mem = TurnMemory(Settings())
    mem.add_user("q")
    mem.reset()
    assert len(mem.snapshot()) == 1