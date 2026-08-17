from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from .audio.codec import AudioChunk


class Segment:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end


class VadEngine(Protocol):
    @property
    def speaking(self) -> bool: ...

    def step(self, chunk: AudioChunk) -> list[str]: ...


class SttEngine(Protocol):
    def transcribe(
        self, audio, sample_rate: int, language: str
    ) -> list[Segment]: ...


class LlmEngine(Protocol):
    async def stream_reply(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]: ...


class TtsEngine(Protocol):
    async def speak(self, texts: AsyncIterator[str]) -> AsyncIterator[AudioChunk]: ...


@dataclass
class PipelineDeps:
    vad: VadEngine
    stt: SttEngine
    llm: LlmEngine
    tts: TtsEngine
    bargein: "BargeinManager"


def build_deps(settings) -> PipelineDeps:
    from .bargein.manager import BargeinManager
    from .llm.client import VllmClient
    from .stt.faster_whisper import FasterWhisperStt
    from .vad.silero import SileroVad

    vad = SileroVad(settings.vad)
    stt = FasterWhisperStt(settings.stt)
    llm = VllmClient(settings.llm)
    if settings.tts.engine == "melo":
        from .tts.melo import MeloEngine

        tts = MeloEngine(settings.tts)
    else:
        from .tts.xtts import XttsEngine

        tts = XttsEngine(settings.tts)
    return PipelineDeps(vad=vad, stt=stt, llm=llm, tts=tts, bargein=BargeinManager())