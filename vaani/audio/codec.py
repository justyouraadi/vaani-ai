import wave
from dataclasses import dataclass
from io import BytesIO

import numpy as np


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    sequence: int = 0


def pcm16_to_chunk(payload: bytes, sample_rate: int, sequence: int = 0) -> AudioChunk:
    samples = np.frombuffer(payload, dtype=np.int16).copy()
    return AudioChunk(samples, sample_rate, sequence)


def np_to_pcm16(samples: np.ndarray) -> bytes:
    return np.ascontiguousarray(samples, dtype=np.int16).tobytes()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    if samples.size == 0:
        return np.zeros(0, dtype=np.int16)
    n_out = int(round(samples.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int16)
    x = np.linspace(0, samples.size - 1, n_out)
    out = np.interp(x, np.arange(samples.size), samples.astype(np.float32))
    return out.astype(np.int16)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.ascontiguousarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(payload), "rb") as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
        channels = w.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples[::channels]
    return samples, rate