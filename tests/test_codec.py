import numpy as np

from vaani.audio import codec


def test_pcm16_roundtrip():
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    payload = codec.np_to_pcm16(samples)
    chunk = codec.pcm16_to_chunk(payload, 16000)
    assert len(payload) == 10
    assert np.array_equal(chunk.samples, samples)
    assert chunk.sample_rate == 16000


def test_resample_16k_to_24k():
    src = np.zeros(1600, dtype=np.int16)
    out = codec.resample(src, 16000, 24000)
    assert out.size == 2400
    assert out.dtype == np.int16


def test_resample_identity():
    src = np.array([1, 2, 3], dtype=np.int16)
    out = codec.resample(src, 16000, 16000)
    assert np.array_equal(out, src)


def test_resample_empty():
    out = codec.resample(np.zeros(0, dtype=np.int16), 16000, 24000)
    assert out.size == 0


def test_wav_roundtrip():
    samples = (np.sin(np.linspace(0, 100, 800)) * 8000).astype(np.int16)
    wav = codec.encode_wav(samples, 16000)
    decoded, rate = codec.decode_wav(wav)
    assert rate == 16000
    assert decoded.size == samples.size
    assert np.array_equal(decoded, samples)