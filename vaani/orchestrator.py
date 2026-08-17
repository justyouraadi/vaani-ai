import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np

from .audio import codec
from .deps import PipelineDeps
from .llm.prompt import TurnMemory
from .settings import Settings
from .tts.chunker import SentenceChunker


@dataclass
class VadEvent:
    kind: str
    chunk: codec.AudioChunk | None = None


class TurnMarker:
    __slots__ = ("cancel_event",)

    def __init__(self, cancel_event: asyncio.Event):
        self.cancel_event = cancel_event


_AUDIO_END = object()


class CallSession:
    def __init__(
        self,
        websocket,
        deps: PipelineDeps,
        settings: Settings,
        call_id: str,
    ):
        self._ws = websocket
        self._deps = deps
        self._settings = settings
        self._call_id = call_id
        self._memory = TurnMemory(settings)
        self._frames_q: asyncio.Queue = asyncio.Queue(128)
        self._speech_q: asyncio.Queue = asyncio.Queue(256)
        self._tts_q: asyncio.Queue = asyncio.Queue(64)
        self._audio_q: asyncio.Queue = asyncio.Queue(32)
        self._cancel_agent = asyncio.Event()
        self._turn_lock = asyncio.Lock()
        self._brain_tasks: set[asyncio.Task] = set()
        self._client_rate = settings.audio.pcm_rate
        self._turns = 0

    async def run(self) -> None:
        workers = [
            asyncio.create_task(self._vad_worker()),
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._tts_worker()),
            asyncio.create_task(self._out_worker()),
        ]
        try:
            await asyncio.wait_for(
                self._reader(), timeout=self._settings.call.timeout_s
            )
        except asyncio.TimeoutError:
            pass
        finally:
            for t in workers + list(self._brain_tasks):
                t.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await asyncio.gather(*self._brain_tasks, return_exceptions=True)

    async def _reader(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("type") == "websocket.receive" and msg.get("bytes"):
                chunk = codec.pcm16_to_chunk(msg["bytes"], self._client_rate)
                await self._frames_q.put(chunk)
            elif msg.get("type") == "websocket.receive" and msg.get("text"):
                await self._on_text(msg["text"])

    async def _on_text(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        mtype = data.get("type")
        if mtype == "config":
            rate = data.get("sample_rate")
            if rate and 8000 <= rate <= 48000:
                self._client_rate = int(rate)
            lang = data.get("language")
            if lang:
                self._settings.stt.language = lang
        elif mtype == "close":
            await self._ws.close(code=1000)
        elif mtype == "reset":
            self._memory.reset()

    async def _vad_worker(self) -> None:
        while True:
            chunk = await self._frames_q.get()
            for event in self._deps.vad.step(chunk):
                if event == "hint":
                    if self._deps.bargein.notify_user_speech():
                        self._stop_agent()
                elif event == "start":
                    await self._speech_q.put(VadEvent("start", chunk))
                elif event == "end":
                    await self._speech_q.put(VadEvent("end"))
            if self._deps.vad.speaking:
                await self._speech_q.put(VadEvent("frame", chunk))

    async def _stt_worker(self) -> None:
        buf = np.zeros(0, dtype=np.float32)
        active = False
        last_partial = 0.0
        while True:
            event = await self._speech_q.get()
            if event.kind == "start":
                active = True
                buf = np.zeros(0, dtype=np.float32)
                last_partial = 0.0
                continue
            if event.kind == "end":
                if active and buf.size:
                    text = await self._transcribe(buf)
                    await self._emit(
                        {"type": "stt_final", "call_id": self._call_id, "text": text}
                    )
                    self._deps.bargein.clear()
                    if text.strip():
                        self._dispatch_turn(text)
                active = False
                buf = np.zeros(0, dtype=np.float32)
                continue
            if not active or event.chunk is None:
                continue
            arr = event.chunk.samples.astype(np.float32) / 32768.0
            buf = np.concatenate([buf, arr])
            now = time.monotonic()
            interval = self._settings.stt.partial_interval_ms / 1000.0
            if now - last_partial >= interval:
                last_partial = now
                partial = await self._transcribe(buf)
                await self._emit(
                    {"type": "stt_partial", "call_id": self._call_id, "text": partial}
                )

    async def _transcribe(self, audio: np.ndarray) -> str:
        segments = await asyncio.to_thread(
            self._deps.stt.transcribe,
            audio,
            self._settings.audio.pcm_rate,
            self._settings.stt.language,
        )
        return "".join(s.text for s in segments).strip()

    def _dispatch_turn(self, text: str) -> None:
        self._turns += 1
        if self._turns > self._settings.call.max_turns:
            asyncio.create_task(
                self._emit({"type": "turn_limit", "call_id": self._call_id})
            )
            return
        task = asyncio.create_task(
            self._brain(text, f"{self._call_id}-{self._turns}")
        )
        self._brain_tasks.add(task)
        task.add_done_callback(self._brain_tasks.discard)

    async def _brain(self, user_text: str, request_id: str) -> None:
        async with self._turn_lock:
            self._stop_agent()
            self._cancel_agent = asyncio.Event()
            marker = TurnMarker(self._cancel_agent)
            await self._tts_q.put(marker)
            messages = self._memory.add_user(user_text)
            await self._emit(
                {
                    "type": "llm_start",
                    "call_id": self._call_id,
                    "turn": self._turns,
                }
            )
            chunker = SentenceChunker(self._settings.tts.max_chunk_chars)
            parts: list[str] = []
            stream = self._deps.llm.stream_reply(messages, request_id)
            try:
                async for token in stream:
                    if self._cancel_agent.is_set():
                        break
                    parts.append(token)
                    for text_chunk in chunker.push(token):
                        if self._cancel_agent.is_set():
                            break
                        await self._tts_q.put(text_chunk)
            finally:
                await stream.aclose()
            if not self._cancel_agent.is_set():
                for text_chunk in chunker.flush():
                    await self._tts_q.put(text_chunk)
                self._memory.add_assistant("".join(parts).strip())
            await self._tts_q.put(marker)
            await self._emit(
                {
                    "type": "llm_end",
                    "call_id": self._call_id,
                    "turn": self._turns,
                    "interrupted": self._cancel_agent.is_set(),
                }
            )

    async def _tts_worker(self) -> None:
        saved: list = []
        while True:
            marker = saved.pop() if saved else await self._tts_q.get()
            if not isinstance(marker, TurnMarker):
                continue
            ev = marker.cancel_event
            if ev.is_set():
                continue
            first_text = await self._tts_q.get()
            if isinstance(first_text, TurnMarker):
                saved.append(first_text)
                continue

            async def gen():
                yield first_text
                while True:
                    item = await self._tts_q.get()
                    if isinstance(item, TurnMarker):
                        saved.append(item)
                        return
                    yield item

            try:
                async for chunk in self._deps.tts.speak(gen()):
                    if ev.is_set():
                        break
                    await self._audio_q.put(chunk)
            except Exception:
                pass
            finally:
                await self._audio_q.put(_AUDIO_END)

    async def _out_worker(self) -> None:
        while True:
            item = await self._audio_q.get()
            if item is _AUDIO_END:
                self._deps.bargein.set_agent_active(False)
                continue
            if self._cancel_agent.is_set():
                continue
            self._deps.bargein.set_agent_active(True)
            out = codec.resample(item.samples, item.sample_rate, self._client_rate)
            try:
                await self._ws.send_bytes(codec.np_to_pcm16(out))
            except Exception:
                return

    def _stop_agent(self) -> None:
        self._cancel_agent.set()
        self._drain(self._tts_q)
        self._drain(self._audio_q)
        self._deps.bargein.set_agent_active(False)

    @staticmethod
    def _drain(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _emit(self, payload: dict) -> None:
        try:
            await self._ws.send_json(payload)
        except Exception:
            pass