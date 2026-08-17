import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets

PCM_RATE = 16000
CHUNK_SAMPLES = 320


def load_pcm16(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples[::channels]
    if rate != PCM_RATE:
        n_out = int(round(len(samples) * PCM_RATE / rate))
        x = np.linspace(0, len(samples) - 1, n_out)
        samples = np.interp(x, np.arange(len(samples)), samples.astype(np.float32)).astype(np.int16)
    return samples


def save_wav(pcm: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(PCM_RATE)
        w.writeframes(pcm)


async def run(url: str, wav_path: Path, out_path: Path, pacing: bool,
              recv_timeout: int, idle_timeout: int) -> None:
    samples = load_pcm16(wav_path)
    total_ms = int(len(samples) * 1000 / PCM_RATE)

    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps({"type": "config", "sample_rate": PCM_RATE}))
        print(f"-> streaming {total_ms} ms of user speech (pacing={'on' if pacing else 'off'})")

        start = time.monotonic()
        for i in range(0, len(samples), CHUNK_SAMPLES):
            chunk = samples[i : i + CHUNK_SAMPLES]
            await ws.send(np.ascontiguousarray(chunk, dtype=np.int16).tobytes())
            if pacing:
                target = (i + len(chunk)) / PCM_RATE
                elapsed = time.monotonic() - start
                if target > elapsed:
                    await asyncio.sleep(target - elapsed)

        print("-> waiting for agent response")
        out = bytearray()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > recv_timeout:
                if not out:
                    print(f"! no response within {recv_timeout}s (is the LLM endpoint reachable?)")
                break
            try:
                msg = await asyncio.wait_for(
                    ws.recv(), timeout=min(recv_timeout - elapsed, idle_timeout)
                )
            except asyncio.TimeoutError:
                if out:
                    print(f"! no audio for {idle_timeout}s, stopping")
                else:
                    print(f"! no response within {recv_timeout}s (is the LLM endpoint reachable?)")
                break
            if isinstance(msg, bytes):
                out += msg
                print(f"  audio +{len(msg):>6} B (total {len(out) / 2000:6.2f} s)")
            else:
                data = json.loads(msg)
                shown = {k: v for k, v in data.items() if k in ("text", "turn", "interrupted")}
                print(f"  event {data.get('type')}: {shown}")

        if out:
            save_wav(bytes(out), out_path)
            print(f"-> agent audio saved to {out_path}")
        else:
            print("-> no agent audio received")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a call against VaaniAI")
    parser.add_argument("--wav", type=Path, default=Path("demo_audio/user_question.wav"))
    parser.add_argument("--out", type=Path, default=Path("demo_audio/agent_reply.wav"))
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws/call")
    parser.add_argument("--no-pacing", action="store_true", help="send all audio instantly")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--idle-timeout", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.wav, args.out, not args.no_pacing,
                    args.timeout, args.idle_timeout))


if __name__ == "__main__":
    main()