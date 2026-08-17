class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._frac = 0;
    this._pending = [];
    this._lastActivity = null;
  }

  process(inputs) {
    const ch = (inputs[0] && inputs[0][0]) || null;
    if (!ch || ch.length === 0) return true;
    const step = sampleRate / 16000;
    let f = this._frac;
    const out = [];
    while (f + 1 < ch.length) {
      const i0 = Math.floor(f);
      const i1 = i0 + 1;
      const t = f - i0;
      const v = ch[i0] * (1 - t) + ch[i1] * t;
      out.push(v);
      f += step;
    }
    this._frac = f - ch.length;
    this._pending.push(...out);
    this._emit();
    return true;
  }

  _emit() {
    while (this._pending.length >= 320) {
      const chunk = new Int16Array(320);
      let sum = 0;
      for (let i = 0; i < 320; i++) {
        const s = this._pending.shift();
        const q = Math.max(-1, Math.min(1, s));
        chunk[i] = q < 0 ? q * 32768 : q * 32767;
        sum += Math.abs(chunk[i]);
      }
      const rms = sum / 320;
      const active = rms > 300;
      if (active !== this._lastActivity) {
        this._lastActivity = active;
        this.port.postMessage({ type: "activity", on: active, rms: Math.round(rms) });
      }
      this.port.postMessage({ type: "pcm16", data: chunk.buffer }, [chunk.buffer]);
    }
  }
}

registerProcessor("pcm16-processor", PCM16Processor);