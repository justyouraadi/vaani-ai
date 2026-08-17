import numpy as np

from ..deps import Segment
from ..settings import SttSettings


class FasterWhisperStt:
    def __init__(self, settings: SttSettings):
        from faster_whisper import WhisperModel

        self._settings = settings
        self._model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root=settings.download_root,
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int, language: str
    ) -> list[Segment]:
        if audio.size == 0:
            return []
        signal = np.asarray(audio, dtype=np.float32)
        if np.abs(signal).max() > 1.0:
            signal = signal / 32768.0
        segments, _ = self._model.transcribe(
            signal,
            language=language or self._settings.language,
            beam_size=2,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return [
            Segment(seg.text.strip(), float(seg.start), float(seg.end))
            for seg in segments
        ]