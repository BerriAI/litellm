import json
import os
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from time import monotonic
from typing import Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger

_FRAME_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_MAX_FRAME_BYTES: Final = 64 * 1024
_SAFE_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,160}")
_MODE: Final = frozenset(("errors", "all"))
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
StreamEnd: TypeAlias = Literal["eof", "closed", "transport_error"]
StreamStage: TypeAlias = Literal["provider_read", "router_emit"]


@dataclass(frozen=True, slots=True)
class _StreamCounts:
    chunks: int = 0
    bytes: int = 0
    events: int = 0
    starts: int = 0
    stops: int = 0
    errors: int = 0
    pings: int = 0
    text_deltas: int = 0
    text_chars: int = 0
    thinking_deltas: int = 0
    tool_deltas: int = 0
    unparsed_frames: int = 0
    first_byte_ms: int | None = None
    first_event_ms: int | None = None
    first_text_ms: int | None = None
    last_byte_ms: int | None = None
    message_id: str | None = None


def _safe_id(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else None


class AnthropicStreamDiagnostics:
    def __init__(
        self,
        mode: str,
        stage: StreamStage,
        call_id: object,
        *,
        upstream_request_id: object = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.mode = mode
        self.stage = stage
        self.call_id = _safe_id(call_id)
        self.upstream_request_id = _safe_id(upstream_request_id)
        self.clock = clock
        self.started = clock()
        self.counts = _StreamCounts()
        self.pending = b""
        self.discarding_frame = False

    @classmethod
    def from_env(
        cls, stage: StreamStage, call_id: object, headers: Mapping[str, str] | None = None
    ) -> "AnthropicStreamDiagnostics | None":
        mode: Final = os.getenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", "off").lower()
        if mode not in _MODE:
            return None
        request_id: Final = next(
            (
                headers.get(name)
                for name in ("request-id", "x-request-id", "x-amzn-requestid", "x-tc-requestid")
                if headers is not None and _safe_id(headers.get(name)) is not None
            ),
            None,
        )
        return cls(mode, stage, call_id, upstream_request_id=request_id)

    def observe(self, chunk: bytes) -> None:
        elapsed: Final = max(0, int((self.clock() - self.started) * 1000))
        self.counts = replace(
            self.counts,
            chunks=self.counts.chunks + 1,
            bytes=self.counts.bytes + len(chunk),
            first_byte_ms=self.counts.first_byte_ms if self.counts.first_byte_ms is not None else elapsed,
            last_byte_ms=elapsed,
        )
        data: Final = self.pending + chunk
        offset = 0  # rebind-ok: scan transport bytes once and retain only a bounded unfinished frame
        for boundary in _FRAME_END.finditer(data):
            if not self.discarding_frame:
                self._observe_frame(data[offset : boundary.start()], elapsed)
            self.discarding_frame = False
            offset = boundary.end()
        tail: Final = data[offset:]
        if len(tail) > _MAX_FRAME_BYTES:
            if not self.discarding_frame:
                self.counts = replace(self.counts, unparsed_frames=self.counts.unparsed_frames + 1)
            self.discarding_frame = True
        self.pending = tail[-3:] if self.discarding_frame else tail

    def _observe_frame(self, frame: bytes, elapsed: int) -> None:
        if len(frame) > _MAX_FRAME_BYTES:
            self.counts = replace(self.counts, unparsed_frames=self.counts.unparsed_frames + 1)
            return
        data: Final = b"\n".join(
            line[len(b"data:") :].removeprefix(b" ") for line in frame.splitlines() if line.startswith(b"data:")
        )
        if not data:
            return
        try:
            payload: Final = _JSON_OBJECT.validate_json(data)
        except (ValueError, UnicodeError):
            self.counts = replace(self.counts, unparsed_frames=self.counts.unparsed_frames + 1)
            return
        event: Final = payload.get("type")
        delta: Final = payload.get("delta")
        delta_type: Final = delta.get("type") if isinstance(delta, dict) and event == "content_block_delta" else None
        text: Final = delta.get("text") if isinstance(delta, dict) and delta_type == "text_delta" else None
        text_chars: Final = len(text) if isinstance(text, str) else 0
        message: Final = payload.get("message")
        message_id: Final = _safe_id(message.get("id")) if isinstance(message, dict) else None
        self.counts = replace(
            self.counts,
            events=self.counts.events + 1,
            starts=self.counts.starts + (event == "message_start"),
            stops=self.counts.stops + (event == "message_stop"),
            errors=self.counts.errors + (event == "error"),
            pings=self.counts.pings + (event == "ping"),
            text_deltas=self.counts.text_deltas + (delta_type == "text_delta"),
            text_chars=self.counts.text_chars + text_chars,
            thinking_deltas=self.counts.thinking_deltas + (delta_type == "thinking_delta"),
            tool_deltas=self.counts.tool_deltas + (delta_type == "input_json_delta"),
            first_event_ms=self.counts.first_event_ms if self.counts.first_event_ms is not None else elapsed,
            first_text_ms=(
                elapsed if text_chars > 0 and self.counts.first_text_ms is None else self.counts.first_text_ms
            ),
            message_id=message_id if event == "message_start" and message_id is not None else self.counts.message_id,
        )

    def finish(self, reason: StreamEnd) -> None:
        complete: Final = (
            self.counts.stops == 1
            and self.counts.errors == 0
            and self.counts.starts == 1
            and not self.pending.strip()
            and not self.discarding_frame
            and self.counts.unparsed_frames == 0
        )
        if self.mode == "errors" and complete and reason != "transport_error":
            return
        summary: Final = {
            "stage": self.stage,
            "call_id": self.call_id,
            "upstream_request_id": self.upstream_request_id,
            "end": reason,
            "complete": complete,
            "elapsed_ms": max(0, int((self.clock() - self.started) * 1000)),
            "pending_bytes": len(self.pending),
            **asdict(self.counts),
        }
        verbose_proxy_logger.info("anthropic_stream_summary %s", json.dumps(summary, separators=(",", ":")))

    async def relay(self, stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        end: StreamEnd = "closed"  # rebind-ok: stream teardown distinguishes EOF, errors and consumer close
        try:
            async for chunk in stream:
                self.observe(chunk)
                yield chunk
            end = "eof"  # rebind-ok: upstream iteration ended normally
        except Exception:
            end = "transport_error"  # rebind-ok: preserve the failure for the final summary
            raise
        finally:
            try:
                await stream.aclose()
            finally:
                self.finish(end)


def finish_stream_diagnostics(diagnostics: AnthropicStreamDiagnostics | None, reason: StreamEnd) -> None:
    if diagnostics is not None:
        diagnostics.finish(reason)
