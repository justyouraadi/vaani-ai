import numpy as np
import torch

from ..audio.codec import AudioChunk, resample
from ..settings import VadSettings

SILERO_RATE = 16000
FRAME_SAMPLES = 512


class SileroVad:
    def __init__(self, settings: VadSettings):
        self._settings = settings
        self._model = self._load_model()
        self._speaking = False
        self._hint_pending = True
        self._speech_run = 0
        self._silence_run = 0
        self._leftover = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _load_model():
        try:
            from silero_vad import load_silero_vad

            return load_silero_vad(model_name="v5")
        except Exception:
            return torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._speaking = False
        self._hint_pending = True
        self._speech_run = 0
        self._silence_run = 0
        self._leftover = np.zeros(0, dtype=np.float32)

    def step(self, chunk: AudioChunk) -> list[str]:
        audio = (
            resample(chunk.samples, chunk.sample_rate, SILERO_RATE).astype(np.float32)
            / 32768.0
        )
        buf = np.concatenate([self._leftover, audio])
        n = len(buf) - len(buf) % FRAME_SAMPLES
        self._leftover = buf[n:]
        events: list[str] = []
        for i in range(0, n, FRAME_SAMPLES):
            prob = self._probability(buf[i : i + FRAME_SAMPLES])
            self._update(prob, events)
        return events

    def _probability(self, frame: np.ndarray) -> float:
        tensor = torch.from_numpy(frame)
        out = self._model(tensor, float(SILERO_RATE))
        return float(out.reshape(-1)[-1])

    def _update(self, prob: float, events: list[str]) -> None:
        cfg = self._settings
        is_speech = prob >= cfg.threshold
        if self._speaking:
            if is_speech:
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._silence_run >= cfg.end_silence_frames:
                    self._speaking = False
                    self._speech_run = 0
                    self._silence_run = 0
                    self._hint_pending = True
                    events.append("end")
            return
        if is_speech:
            self._speech_run += 1
            if self._hint_pending and self._speech_run >= cfg.hint_frames:
                self._hint_pending = False
                events.append("hint")
            if self._speech_run >= cfg.start_frames:
                self._speaking = True
                self._silence_run = 0
                events.append("start")
        else:
            self._speech_run = 0