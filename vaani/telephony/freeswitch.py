import asyncio
import base64
import json
from typing import Optional


class FreeSwitchForkClient:
    def __init__(
        self,
        host: str,
        port: int,
        session_uuid: str,
        codec: str = "L16",
        sample_rate: int = 8000,
    ):
        self._host = host
        self._port = port
        self._session_uuid = session_uuid
        self._codec = codec
        self._sample_rate = sample_rate
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )
        await self._send(
            {
                "command": "connect",
                "session_uuid": self._session_uuid,
                "codec": self._codec,
                "rate": self._sample_rate,
                "channels": 1,
            }
        )

    async def _send(self, payload: dict) -> None:
        line = json.dumps(payload) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()

    async def recv_audio(self) -> dict:
        line = await self._reader.readline()
        if not line:
            return {}
        return json.loads(line)

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send(
            {
                "command": "send_audio",
                "session_uuid": self._session_uuid,
                "base64": base64.b64encode(pcm16).decode(),
            }
        )

    async def playback_start(self) -> None:
        await self._send(
            {"command": "playback_start", "session_uuid": self._session_uuid}
        )

    async def playback_stop(self) -> None:
        await self._send(
            {"command": "playback_stop", "session_uuid": self._session_uuid}
        )

    async def close(self) -> None:
        if self._writer:
            try:
                await self._send({"command": "exit"})
            except Exception:
                pass
            self._writer.close()
            await self._writer.wait_closed()