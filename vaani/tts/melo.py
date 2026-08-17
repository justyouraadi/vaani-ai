import asyncio
from typing import AsyncIterator

import numpy as np

from ..audio.codec import AudioChunk
from ..settings import TtsSettings

MELO_RATE = 44100


class MeloEngine:
    def __init__(self, settings: TtsSettings):
        from melo_tts import MeloTTS

        self._settings = settings
        self._model = MeloTTS(
            language=settings.language,
            device=settings.device,
        )

    async def speak(self, texts: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        seq = 0
        async for text in texts:
            arr = await asyncio.to_thread(self._model.tts, text)
            samples = np.asarray(arr, dtype=np.int16)
            if samples.size == 0:
                continue
            yield AudioChunk(samples, MELO_RATE, seq)
            seq += 1