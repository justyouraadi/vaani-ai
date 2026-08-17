END_MARKS = ".,!?;।"


class SentenceChunker:
    def __init__(self, max_chars: int = 240):
        self._max = max_chars
        self._buf = ""
        self._first_emitted = False

    def push(self, token: str) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        if not self._first_emitted:
            self._first_emitted = True
            if token.strip():
                out.append((token, False))
            return out
        self._buf += token
        self._split_until_done(out)
        return out

    def flush(self) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        if self._buf.strip():
            out.append((self._buf.strip(), True))
        self._buf = ""
        self._first_emitted = False
        return out

    def _split_until_done(self, out: list[tuple[str, bool]]) -> None:
        while True:
            if len(self._buf) <= self._max:
                idx = self._find_end_mark()
                if idx >= 0:
                    chunk = self._buf[: idx + 1]
                    self._buf = self._buf[idx + 1 :]
                    out.append((chunk.strip(), True))
                    continue
                return
            cut = self._cut_at(self._max)
            chunk = self._buf[:cut]
            self._buf = self._buf[cut:]
            out.append((chunk.strip(), False))

    def _find_end_mark(self) -> int:
        for i, ch in enumerate(self._buf):
            if ch in END_MARKS:
                return i
        return -1

    def _cut_at(self, limit: int) -> int:
        s = self._buf.rfind(" ", 0, limit)
        return s if s > 0 else limit