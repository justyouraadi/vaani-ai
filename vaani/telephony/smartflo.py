import json
from typing import Optional

from .protocol import WebSocketMediaTransport


class SmartFloClient:
    def __init__(
        self,
        url: str,
        token: str,
        call_id: str,
        sample_rate: int = 16000,
    ):
        self._url = url
        self._token = token
        self._call_id = call_id
        self._sample_rate = sample_rate

    async def connect(self) -> WebSocketMediaTransport:
        import websockets

        ws = await websockets.connect(
            self._url,
            additional_headers={
                "Authorization": f"Bearer {self._token}",
                "X-Call-Id": self._call_id,
            },
            max_size=2**24,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "session_start",
                    "call_id": self._call_id,
                    "sample_rate": self._sample_rate,
                    "codec": "PCM16",
                    "barge_in": True,
                }
            )
        )
        transport = WebSocketMediaTransport(ws, self._sample_rate)
        return transport

    async def hangup(self, transport: Optional[WebSocketMediaTransport]) -> None:
        if transport is None:
            return
        try:
            await transport._ws.send(json.dumps({"type": "session_end"}))
        except Exception:
            pass
        await transport.close()