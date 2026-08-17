import asyncio
import queue
from typing import AsyncIterator

import numpy as np

from ..audio.codec import AudioChunk
from ..settings import TtsSettings

XTTS_RATE = 24000


class XttsEngine:
    def __init__(self, settings: TtsSettings):
        from TTS.api import TTS

        self._settings = settings
        self._api = TTS(
            model_path=settings.model_path,
            gpu=settings.device == "cuda",
        )

    async def speak(self, texts: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        text_q: queue.Queue = queue.Queue()
        audio_q: queue.Queue = queue.Queue()

        async def feed() -> None:
            try:
                async for t in texts:
                    await loop.run_in_executor(None, text_q.put, t)
            finally:
                await loop.run_in_executor(None, text_q.put, None)

        def synthesize() -> None:
            def gen():
                while True:
                    t = text_q.get()
                    if t is None:
                        return
                    yield t

            try:
                for arr in self._api.tts(
                    text=gen(),
                    speaker_wav=self._settings.speaker_wav,
                    language=self._settings.language,
                    stream=True,
                    stream_chunk_size=self._settings.stream_chunk_size,
                    add_wav_header=False,
                ):
                    audio_q.put(arr)
            finally:
                audio_q.put(None)

        feed_task = asyncio.create_task(feed())
        synth_task = loop.run_in_executor(None, synthesize)
        seq = 0
        try:
            while True:
                arr = await loop.run_in_executor(None, audio_q.get)
                if arr is None:
                    return
                samples = np.asarray(arr, dtype=np.int16)
                if samples.size == 0:
                    continue
                yield AudioChunk(samples, XTTS_RATE, seq)
                seq += 1
        finally:
            feed_task.cancel()
            await asyncio.gather(feed_task, return_exceptions=True)
            loop.run_in_executor(None, audio_q.put, None)
            if synth_task.running():
                await asyncio.wait_for(
                    asyncio.shield(asyncio.wrap_future(synth_task)), timeout=5
                )