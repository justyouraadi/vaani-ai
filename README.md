# TBB-VaaniAI

Self-hosted, ultra-low-latency (sub-300ms) real-time voice AI agent for Hindi/Hinglish, built on a fully streaming pipeline with barge-in. Raw FastAPI + WebSockets — no framework lock-in.

## Architecture

```
                    ┌─────────────────────── Server B (edge) ───────────────────────┐
Callee/Caller ──WebSocket──► FastAPI orchestrator                                  │
   │ (PCM16 16kHz)           │                                                     │
   ▼                         ▼                                                     │
frames ──► Silero VAD ──hint──► Barge-in Manager ──kill──► TTS sink (drops audio)  │
   │        (32ms frames)     (12-140ms kill signal)                               │
   ▼                                                                            │
   └► Faster-Whisper large-v3-turbo (200ms chunk streaming, partials every 700ms)│
        │  final transcript                                                       │
        ▼                                                                         │
   TurnMemory + filler-word Hinglish persona prompt                               │
        ▼                                                                         │
   vLLM (Qwen2.5-14B) ──OpenAI-compatible SSE token stream──► SentenceChunker      │
        │                                               │ (punctuation-delimited) │
        └── tokens stream live ──► filler sentence flushes to TTS IMMEDIATELY     │
                                                     ▼                             │
                                      XTTS v2 (stream=True, voice-cloned Vaani)   │
                                                     ▼                             │
                                      resample → PCM16 → WebSocket → caller      │
                     └──────────── Server A (brain): vLLM + GPU ───────────────┘
```

Every stage is an independent `asyncio` worker connected by bounded queues
(`vaani/orchestrator.py`). Nothing waits for the stage before it:

- LLM first token (filler word like `Hmm, `) → chunker flushes it to TTS instantly while the LLM keeps generating.
- TTS synthesizes sentences incrementally through one streaming `speak()` session per turn (`TurnMarker` protocol in `orchestrator.py`).
- User speech during playback → VAD `hint` event (fires ~128ms in) → kill signal: current turn's audio and LLM generation are cancelled, the new utterance takes over.

## Repository layout

```
vaani/
  main.py            FastAPI app; /ws/call endpoint; lazy GPU model load
  orchestrator.py    CallSession: reader, VAD/STT/TTS/out workers + barge-in
  settings.py        YAML + env-driven configuration dataclasses
  deps.py            engine protocols + GPU dependency builder
  vad/silero.py      Silero VAD v5 state machine (hint/start/end events)
  stt/faster_whisper.py  streaming windowed transcription (partials + final)
  llm/client.py      vLLM SSE streaming client (OpenAI-compatible API)
  llm/prompt.py      Hinglish persona prompt + filler-word instruction, TurnMemory
  tts/chunker.py     punctuation sentence chunker (also splits on danda ।)
  tts/xtts.py        XTTS v2 streaming engine (voice cloning, live text feed)
  tts/melo.py        MeloTTS fallback engine (CPU-friendly)
  bargein/manager.py interrupt state machine
  audio/codec.py     PCM16/resample/WAV helpers
  telephony/         Tata Smartflo + FreeSWITCH mod_audio_fork adapters
scripts/
  download_models.py faster-whisper + XTTS weights
  warm_kv_cache.py   pin the persona prefix in vLLM's KV cache
deploy/              Dockerfile, docker-compose.brain (vLLM) + edge (STT/TTS/API)
tests/               unit + full WebSocket integration tests with fake engines
```

## Quickstart (two GPU servers)

### Server A — the brain (vLLM)

```bash
make brain            # deploy/docker-compose.brain.yml: Qwen2.5-14B w/ prefix caching
python scripts/warm_kv_cache.py --base-url http://localhost:8000/v1
```

`--enable-prefix-caching` is what makes the fixed system prompt (persona) near-free
on every call after warmup.

### Server B — edge (STT/TTS/API)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install coqui-tts            # XTTS v2
make models                      # whisper large-v3-turbo + XTTS-v2 weights
cp my_voice_sample.wav models/vaani.wav   # 10-30s clean Hindi clip for cloning
cp .env.example .env && set VAANI_LLM_BASE_URL=http://<server-a>:8000/v1
make dev                         # uvicorn on :8765
```

## RunPod live deployment (single A40 48 GB pod — full 14B stack)

Your exact hardware fits everything on ONE pod, all on the persistent volume
(`/workspace`), so pod restarts never lose models. Skip the two-server split.

**Pod**: 1× A40 (48 GB VRAM) — RunPod template
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. VRAM budget:
Qwen2.5-14B ≈ 28 GB + KV ≈ 5 GB (vLLM capped at 0.72 util), whisper
large-v3-turbo ≈ 2 GB, XTTS v2 ≈ 3 GB — ~40 GB peak, fits comfortably.

1. **Ship the repo** (target the persistent volume; the rsync keeps
   `models/vaani.wav` but skips local junk):
   ```bash
   rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
     --exclude 'demo_audio' --include 'models/vaani.wav' --exclude 'models/*' \
     . root@<pod-ip>:/workspace/vaani/
   ```

2. **Deploy everything in one shot** (installs vllm 0.7.3 + torch 2.5.1 cu124,
   downloads weights into the pod's persistent `models/`, starts brain :8000
   and edge :8765):
   ```bash
   ssh root@<pod-ip> "cd /workspace/vaani && bash deploy/setup_runpod.sh all"
   ssh root@<pod-ip> "tail -f /workspace/logs/vaani-brain.log"   # wait for 'Application startup complete'
   ssh root@<pod-ip> "tail -f /workspace/logs/vaani-edge.log"
   ```

3. **Test live from your laptop** over SSH tunnels (pods expose all ports;
   prefer tunnels):
   ```bash
   ssh -N -L 8000:127.0.0.1:8000 -L 8765:127.0.0.1:8765 root@<pod-ip> &
   curl -s localhost:8000/v1/models && curl -s localhost:8765/health
   .venv/bin/python scripts/warm_kv_cache.py --base-url http://localhost:8000/v1
   make demo                          # real call: VAD→STT→LLM→XTTS→clone audio
   afplay demo_audio/agent_reply.wav  # hear the clone answer
   ```

4. **Check VRAM split & restart behavior**:
   `nvidia-smi` on the pod — vLLM should sit near 0.72×48 GB ≈ 35 GB, edge
   engines load lazily on first call. If anything OOMs, drop the model:
   `VAANI_MODEL=Qwen/Qwen2.5-7B-Instruct bash deploy/setup_runpod.sh all`
   (after killing the running vLLM process).
   After a pod restart, re-run step 2 — the script is idempotent and resumes
   your exact setup.

All routes stay the same for real telephony: point
`scripts/test_call.py --url ws://<pod-ip>:8765/ws/call` at the public IP and
pipe a live call recording through it (plus `--api-key` on vLLM or a firewall
rule if you open ports publicly).

## WebSocket protocol (`/ws/call`)

| Direction | Type | Payload |
|---|---|---|
| client → server | binary | PCM16 LE mono, 16 kHz (set via config) |
| client → server | text | `{"type":"config","sample_rate":16000,"language":"hi"}` |
| client → server | text | `{"type":"close"}` |
| server → client | binary | PCM16 LE mono, 16 kHz synthesized speech |
| server → client | text | `{"type":"stt_partial"\|"stt_final","text":...}` |
| server → client | text | `{"type":"llm_start"\|"llm_end","turn":N,"interrupted":bool}` |

A `telephony/*` adapter (Smartflo/FreeSWITCH) on the operator side converts SIP
media into this wire format.

### Live browser demo UI

The edge server also serves a real-time demo page (no WAV files, no CLI):
`http://<edge-host>:8765/demo/` (or `http://localhost:8765/`). Built with
`AudioWorklet` (`vaani/demo/pcm16-processor.js`): mic audio is downsampled to
PCM16 16 kHz on-device and streamed over the same `/ws/call` socket; Vaani's
replies play back through a gapless buffer queue. Client-side VAD activity
detection stops playback the moment you speak into the reply — the same
barge-in you feel on a phone call, with the live transcript and pipeline events
in the console panel.

## Latency playbook — where each trick lives

| Trick | Implementation |
|---|---|
| Filler-word prompting | `llm/prompt.py` — system prompt forces `Hmm.../Ji.../Achha...` first; first token flushes through `tts/chunker.py` immediately |
| Sentence chunking | `tts/chunker.py` — emits on `.,!?;।`, hard-splits at 240 chars |
| KV-cache pre-warm | `scripts/warm_kv_cache.py` + `--enable-prefix-caching` on vLLM |
| Barge-in kill | `vad/silero.py` hint event (~128 ms) → `bargein/manager.py` → orchestrator kills LLM stream + TTS sink |
| Parallel stages | four bounded-queue workers in `orchestrator.py`; LLM and TTS never wait on each other |
| Grouped synthesis | one XTTS `speak()` session per turn with a live text generator — sentence 2 starts while sentence 1 plays |

### Tuning knobs (`config/settings.yaml`)

- `vad.hint_frames` — how fast barge-in fires (4 × 32 ms = 128 ms). Lower = snappier, slightly noisier.
- `vad.end_silence_frames` — end-of-utterance commit (22 × 32 ms ≈ 700 ms). Lower = faster replies, risk of cutting mid-word.
- `stt.partial_interval_ms` — live transcript cadence (UI/telemetry only; replies trigger on final).
- `llm.max_tokens` / `temperature` — spoken answers should be short; 256 tokens ≈ 20-30 s of speech.
- `tts.stream_chunk_size` — XTTS chunk granularity; 16 is a good latency/quality balance.

## Testing

```bash
make test
```

23 tests, all engine fakes — no GPU required. The barge-in integration test
(`tests/test_orchestrator.py::test_barge_in_interrupts_agent_mid_playback`)
proves: agent is mid-playback → user speaks → turn 1 LLM stream is killed
(`llm_end.interrupted=true`), no leaked audio, turn 2 plays.

### End-to-end simulated call (local, no GPU)

Streams a real audio file over the actual WebSocket protocol and saves the
agent's reply:

```bash
make dev-local demo-voice demo
```

1. `make dev-local` — real VAD/STT/TTS on CPU + Ollama as the LLM (`config/settings.local.yaml`).
   Requires `pip install faster-whisper melo-tts` in the venv, plus [Ollama](https://ollama.com)
   (`ollama pull qwen2.5:7b`) for the OpenAI-compatible endpoint.
2. `make demo-voice` — synthesizes `demo_audio/user_question.wav` (macOS `say`).
3. `make demo` — `scripts/test_call.py` paces the audio over `ws://127.0.0.1:8765/ws/call`
   in real time, prints `stt_*`/`llm_*` events, saves `demo_audio/agent_reply.wav`.

`scripts/test_call.py` is also your telephony/gateway smoke test — point `--url`
at a deployed brain and pipe a real customer recording through it.
(`--no-pacing` streams instantly for fast iteration.)

## Notes

- Telephony adapters are protocol skeletons; confirm the exact framing with your
  gateway (Smartflo ships a JSON+base64-audio framing; FreeSWITCH uses
  `mod_audio_fork` JSON-lines).
- All models self-hosted → no per-minute API costs, no audio leaves your data
  center.
- If you prefer a framework, the orchestrator is drop-in Pipecat-compatible at
  the interface level (`VadEngine/SttEngine/LlmEngine/TtsEngine` protocols).