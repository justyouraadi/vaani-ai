import base64
import json
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class MediaFrame:
    sample_rate: int
    pcm16: bytes
    metadata: Optional[dict] = None


class MediaTransport(Protocol):
    async def send_audio(self, pcm16: bytes) -> None: ...

    async def recv(self) -> Optional[MediaFrame]: ...

    async def close(self) -> None: ...


class WebSocketMediaTransport:
    def __init__(self, websocket, sample_rate: int = 16000):
        self._ws = websocket
        self._rate = sample_rate

    async def send_audio(self, pcm16: bytes) -> None:
        await self._ws.send_bytes(pcm16)

    async def recv(self) -> Optional[MediaFrame]:
        msg = await self._ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return None
        if msg.get("type") == "websocket.receive":
            if msg.get("bytes"):
                return MediaFrame(self._rate, msg["bytes"])
            if msg.get("text"):
                return self._parse_text(msg["text"])
        return None

    def _parse_text(self, raw: str) -> Optional[MediaFrame]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and data.get("audio"):
            pcm = base64.b64decode(data["audio"])
            rate = int(data.get("sample_rate", self._rate))
            return MediaFrame(rate, pcm, data)
        return MediaFrame(self._rate, b"", data if isinstance(data, dict) else None)

    async def close(self) -> None:
        try:
            await self._ws.close(code=1000)
        except Exception:
            pass