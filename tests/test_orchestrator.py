import asyncio
import json

import numpy as np
from fastapi.testclient import TestClient

from vaani.audio.codec import AudioChunk
from vaani.bargein.manager import BargeinManager
from vaani.deps import PipelineDeps, Segment
from vaani.main import create_app
from vaani.settings import Settings

PCM_FRAME = b"\x00\x00" * 160


def _drain_to(ws, need_finals=1, need_audio=0, need_interrupted=False, need_llm_start=0, limit=100):
    texts, audio = [], []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("type") != "websocket.send":
            continue
        if msg.get("bytes"):
            audio.append(msg["bytes"])
        elif msg.get("text"):
            texts.append(json.loads(msg["text"]))
        finals = sum(1 for t in texts if t.get("type") == "stt_final")
        starts = sum(1 for t in texts if t.get("type") == "llm_start")
        interrupted = any(
            t.get("type") == "llm_end" and t.get("interrupted")
            for t in texts
        )
        if finals >= need_finals and len(audio) >= need_audio and starts >= need_llm_start:
            if not need_interrupted or interrupted:
                return texts, audio
    return texts, audio


class FakeVad:
    def __init__(self):
        self._n = 0
        self._speaking = False

    @property
    def speaking(self):
        return self._speaking

    def step(self, chunk):
        self._n += 1
        out = []
        if self._n == 1:
            self._speaking = True
            out.append("start")
        elif self._n == 3:
            self._speaking = False
            out.append("end")
        return out


class FakeStt:
    def transcribe(self, audio, sample_rate, language):
        if audio.size == 0:
            return []
        return [Segment("namaste", 0.0, 0.5)]


class FakeLlm:
    TOKENS = ["Hmm, ", "namaste ", "main ", "vaani ", "hoon!"]

    async def stream_reply(self, messages, request_id):
        for t in self.TOKENS:
            yield t


class FakeTts:
    async def speak(self, texts):
        async for _text in texts:
            yield AudioChunk(np.zeros(2400, dtype=np.int16), 24000)


class TwoTurnVad:
    def __init__(self):
        self._n = 0
        self._speaking = False

    @property
    def speaking(self):
        return self._speaking

    def step(self, chunk):
        self._n += 1
        out = []
        if self._n == 1:
            self._speaking = True
            out.append("start")
        elif self._n == 3:
            self._speaking = False
            out.append("end")
        elif self._n == 5:
            self._speaking = True
            out.extend(["hint", "start"])
        elif self._n == 7:
            self._speaking = False
            out.append("end")
        return out


class SlowLlm(FakeLlm):
    async def stream_reply(self, messages, request_id):
        for t in self.TOKENS:
            await asyncio.sleep(0.15)
            yield t


def make_client(vad=None, tts=None):
    deps = PipelineDeps(
        vad=vad or FakeVad(),
        stt=FakeStt(),
        llm=FakeLlm(),
        tts=tts or FakeTts(),
        bargein=BargeinManager(),
    )
    app = create_app(settings=Settings(), deps=deps)
    return TestClient(app)


def test_full_call_flow():
    client = make_client()
    with client.websocket_connect("/ws/call?call_id=test-1") as ws:
        for _ in range(4):
            ws.send_bytes(PCM_FRAME)
        texts, audio = _drain_to(ws, need_finals=1, need_audio=1)
        types = [t.get("type") for t in texts]
        assert "stt_final" in types
        final = next(t for t in texts if t.get("type") == "stt_final")
        assert final["text"] == "namaste"
        assert "llm_start" in types
        assert len(audio) > 0
        assert len(audio[-1]) % 2 == 0
        ws.close()


def test_barge_in_interrupts_agent_mid_playback():
    deps = PipelineDeps(
        vad=TwoTurnVad(),
        stt=FakeStt(),
        llm=SlowLlm(),
        tts=FakeTts(),
        bargein=BargeinManager(),
    )
    client = TestClient(create_app(settings=Settings(), deps=deps))
    with client.websocket_connect("/ws/call") as ws:
        for _ in range(4):
            ws.send_bytes(PCM_FRAME)
        texts, audio = _drain_to(ws, need_finals=1, need_audio=1)
        assert "llm_start" in [t.get("type") for t in texts]
        assert len(audio) == 1, "agent should be mid-playback before interrupt"

        for _ in range(3):
            ws.send_bytes(PCM_FRAME)
        texts, audio2 = _drain_to(
            ws, need_finals=1, need_audio=0, need_interrupted=True,
            need_llm_start=1,
        )
        types = [t.get("type") for t in texts]
        assert "stt_final" in types, "turn 2 never finalized"
        assert "llm_start" in types, "turn 2 never started"
        assert any(
            t.get("type") == "llm_end" and t.get("interrupted") for t in texts
        ), "barge-in did not kill turn 1"
        assert len(audio2) == 0, "interrupted turn leaked audio"

        texts, audio3 = _drain_to(ws, need_finals=0, need_audio=1)
        assert len(audio3) >= 1, "turn 2 audio never played after interrupt"
        ws.close()


def test_ws_accepts_config_and_close():
    client = make_client()
    with client.websocket_connect("/ws/call") as ws:
        ws.send_text(json.dumps({"type": "config", "sample_rate": 8000}))
        ws.send_text(json.dumps({"type": "close"}))
    assert True