"""Strip non-Anthropic SSE control noise from outbound /v1/messages streams.

Some upstream providers (notably OpenRouter) inject SSE events that are not part
of the Anthropic SSE spec into the stream they emit for Anthropic-format
requests:
  - SSE comment keep-alives like `: OPENROUTER PROCESSING`
  - An OpenAI-style `data: [DONE]` terminator
  - `data:` lines whose payload is empty or itself a `:`-prefixed comment

Strict Anthropic SDK clients and downstream LiteLLM proxies that JSON-parse every
`data:` line treat these as malformed input. This filter drops them while
preserving valid Anthropic events across arbitrary `aiter_bytes()` chunk
boundaries.
"""

from typing import List, Tuple


class AnthropicSSENoiseFilter:
    """Stateful, chunk-boundary-safe filter for Anthropic SSE streams.

    Usage:
        f = AnthropicSSENoiseFilter()
        async for chunk in upstream.aiter_bytes():
            out = f.feed(chunk)
            if out:
                yield out
        tail = f.flush()
        if tail:
            yield tail
    """

    def __init__(self) -> None:
        self._buf: str = ""
        self._undecoded: bytes = b""

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        data = self._undecoded + chunk
        try:
            self._buf += data.decode("utf-8")
            self._undecoded = b""
        except UnicodeDecodeError as exc:
            split = exc.start
            self._buf += data[:split].decode("utf-8")
            self._undecoded = data[split:]

        emit_parts: List[str] = []
        while True:
            sep_idx, sep_len = self._find_separator(self._buf)
            if sep_idx == -1:
                break
            event = self._buf[:sep_idx]
            sep = self._buf[sep_idx : sep_idx + sep_len]
            self._buf = self._buf[sep_idx + sep_len :]
            if not self._is_noise_event(event):
                emit_parts.append(event + sep)
        return "".join(emit_parts).encode("utf-8")

    def flush(self) -> bytes:
        leftover = self._buf + self._undecoded.decode("utf-8", errors="replace")
        self._buf = ""
        self._undecoded = b""
        if not leftover or self._is_noise_event(leftover):
            return b""
        return leftover.encode("utf-8")

    @staticmethod
    def _find_separator(s: str) -> Tuple[int, int]:
        idx_lf = s.find("\n\n")
        idx_crlf = s.find("\r\n\r\n")
        if idx_lf == -1 and idx_crlf == -1:
            return -1, 0
        if idx_crlf == -1 or (idx_lf != -1 and idx_lf < idx_crlf):
            return idx_lf, 2
        return idx_crlf, 4

    @staticmethod
    def _is_noise_event(event: str) -> bool:
        for ln in event.splitlines():
            stripped = ln.strip()
            if not stripped.startswith("data:"):
                continue
            payload = stripped[5:].strip()
            if payload and payload != "[DONE]" and not payload.startswith(":"):
                return False
        return True
