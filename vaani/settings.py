import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Persona:
    name: str = "Vaani"
    language: str = "Hindi/Hinglish"
    role: str = (
        "You are Vaani, a warm, patient, and helpful voice assistant for a "
        "Hindi-speaking caller. Be polite, use spoken-conversation "
        "Hindi/Hinglish, keep sentences short, never use markdown, never "
        "spell out numbers, and never mention that you are an AI."
    )
    filler_words: list[str] = field(
        default_factory=lambda: ["Hmm", "Ji", "Achha", "Dekhiye"]
    )


@dataclass
class VadSettings:
    sample_rate: int = 16000
    frame_ms: int = 32
    threshold: float = 0.5
    hint_frames: int = 4
    start_frames: int = 8
    end_silence_frames: int = 22


@dataclass
class SttSettings:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "hi"
    download_root: str = "models/whisper"
    partial_interval_ms: int = 700


@dataclass
class LlmSettings:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 256
    fillers: bool = True
    extra_stop: list[str] = field(default_factory=list)


@dataclass
class TtsSettings:
    engine: str = "xtts"
    model_path: str = "models/xtts-v2"
    speaker_wav: str = "models/vaani.wav"
    language: str = "hi"
    stream_chunk_size: int = 16
    max_chunk_chars: int = 240
    device: str = "cuda"


@dataclass
class AudioSettings:
    pcm_rate: int = 16000
    tts_rate: int = 24000


@dataclass
class CallSettings:
    timeout_s: int = 900
    max_turns: int = 12


@dataclass
class Settings:
    call: CallSettings = field(default_factory=CallSettings)
    persona: Persona = field(default_factory=Persona)
    vad: VadSettings = field(default_factory=VadSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        path = os.environ.get(
            "VAANI_SETTINGS",
            str(Path(__file__).resolve().parent.parent / "config" / "settings.yaml"),
        )
    path = Path(path)
    settings = Settings()
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        settings.persona = Persona(**(raw.get("persona") or {}))
        settings.vad = VadSettings(**(raw.get("vad") or {}))
        settings.stt = SttSettings(**(raw.get("stt") or {}))
        settings.llm = LlmSettings(**(raw.get("llm") or {}))
        settings.tts = TtsSettings(**(raw.get("tts") or {}))
        settings.audio = AudioSettings(**(raw.get("audio") or {}))
        settings.call = CallSettings(**(raw.get("call") or {}))
    settings.llm.base_url = os.environ.get("VAANI_LLM_BASE_URL", settings.llm.base_url)
    settings.llm.model = os.environ.get("VAANI_LLM_MODEL", settings.llm.model)
    settings.stt.model = os.environ.get("VAANI_STT_MODEL", settings.stt.model)
    settings.tts.engine = os.environ.get("VAANI_TTS_ENGINE", settings.tts.engine)
    settings.tts.speaker_wav = os.environ.get(
        "VAANI_SPEAKER_WAV", settings.tts.speaker_wav
    )
    return settings