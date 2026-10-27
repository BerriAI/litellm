import json
import os
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from time import monotonic, time
from typing import Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger

_FRAME_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_MAX_FRAME_BYTES: Final = 64 * 1024
_SAFE_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,160}")
_MODE: Final = frozenset(("errors", "slow", "all"))
_EVENTS: Final = frozenset(
    (
        "message_start",
        "message_stop",
        "message_delta",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    )
)
_REQUEST_ID_HEADERS: Final = (
    "x-tc-requestid",
    "x-tc-request-id",
    "x-tencent-request-id",
    "request-id",
    "x-request-id",
    "x-amzn-requestid",
)
_SLOW_CONTENT_MS: Final = 30_000
_SLOW_SEND_MS: Final = 1_000
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
StreamEnd: TypeAlias = Literal["eof", "closed", "transport_error"]
StreamStage: TypeAlias = Literal["provider_read", "router_emit", "http_send"]


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
    empty_data_frames: int = 0
    first_byte_ms: int | None = None
    first_event_ms: int | None = None
    first_text_ms: int | None = None
    first_content_ms: int | None = None
    last_content_ms: int | None = None
    max_content_gap_ms: int = 0
    first_event_type: str | None = None
    last_event_type: str | None = None
    last_byte_ms: int | None = None
    message_id: str | None = None


def _safe_id(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) and not value.startswith("sk-") else None


class AnthropicStreamDiagnostics:
    def __init__(
        self,
        mode: str,
        stage: StreamStage,
        call_id: object,
        *,
        upstream_request_id: object = None,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        self.mode = mode
        self.stage = stage
        self.call_id = _safe_id(call_id)
        self.upstream_request_id = _safe_id(upstream_request_id)
        self.clock = clock
        self.started = clock()
        self.started_at_unix_ms = int(wall_clock() * 1000)
        self.counts = _StreamCounts()
        self.pending = b""
        self.discarding_frame = False
        self.finished = False
        self.headers_sent_ms: int | None = None
        self.send_started_ms: int | None = None
        self.first_send_attempt_ms: int | None = None
        self.last_send_completed_ms: int | None = None
        self.max_send_ms = 0
        self.send_attempts = 0
        self.sends_completed = 0
        self.send_failed = False
        self.http_complete = False

    def elapsed_ms(self) -> int:
        return max(0, int((self.clock() - self.started) * 1000))

    def start_send(self) -> None:
        self.send_started_ms = self.elapsed_ms()
        self.send_attempts += 1
        if self.first_send_attempt_ms is None:
            self.first_send_attempt_ms = self.send_started_ms

    def complete_send(self, chunk: bytes) -> None:
        self.last_send_completed_ms = self.elapsed_ms()
        if self.send_started_ms is not None:
            self.max_send_ms = max(self.max_send_ms, self.last_send_completed_ms - self.send_started_ms)
        self.send_started_ms = None
        self.sends_completed += 1
        self.observe(chunk)

    @classmethod
    def from_env(
        cls, stage: StreamStage, call_id: object, headers: Mapping[str, str] | None = None
    ) -> "AnthropicStreamDiagnostics | None":
        mode: Final = os.getenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", "off").lower()
        if mode not in _MODE:
            return None
        normalized_headers: Final = {name.lower(): value for name, value in (headers or {}).items()}
        request_id: Final = next(
            (
                normalized_headers.get(name)
                for name in (
                    _REQUEST_ID_HEADERS
                    if stage == "provider_read"
                    else tuple("llm_provider-" + key for key in _REQUEST_ID_HEADERS)
                )
                if _safe_id(normalized_headers.get(name)) is not None
            ),
            None,
        )
        return cls(mode, stage, call_id, upstream_request_id=request_id)

    def observe(self, chunk: bytes) -> None:
        if self.finished or not chunk:
            return
        elapsed: Final = self.elapsed_ms()
        self.counts = replace(
            self.counts,
            chunks=self.counts.chunks + 1,
            bytes=self.counts.bytes + len(chunk),
            first_byte_ms=self.counts.first_byte_ms if self.counts.first_byte_ms is not None else elapsed,
            last_byte_ms=elapsed,
        )
        data: Final = self.pending + chunk
        offset = 0  # rebind-ok: scan transport bytes once and retain only a bounded unfinished frame
        for boundary in _FRAME_END.finditer(data, max(0, len(self.pending) - 3)):
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
        data_lines: Final = tuple(
            line[len(b"data:") :].removeprefix(b" ")
            for line in frame.splitlines()
            if line.startswith(b"data:") or line == b"data"
        )
        if not data_lines:
            return
        data: Final = b"\n".join(data_lines)
        if not data.strip():
            self.counts = replace(self.counts, empty_data_frames=self.counts.empty_data_frames + 1)
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
        content: Final = (
            delta.get(
                {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}.get(
                    delta_type, ""
                )
            )
            if isinstance(delta, dict) and isinstance(delta_type, str)
            else None
        )
        has_content: Final = isinstance(content, str) and bool(content)
        event_name: Final = event if isinstance(event, str) and event in _EVENTS else "other"
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
            first_content_ms=elapsed
            if has_content and self.counts.first_content_ms is None
            else self.counts.first_content_ms,
            last_content_ms=elapsed if has_content else self.counts.last_content_ms,
            max_content_gap_ms=max(self.counts.max_content_gap_ms, elapsed - (self.counts.last_content_ms or 0))
            if has_content
            else self.counts.max_content_gap_ms,
            first_event_type=self.counts.first_event_type or event_name,
            last_event_type=event_name,
            message_id=message_id if event == "message_start" and message_id is not None else self.counts.message_id,
        )

    def finish(self, reason: StreamEnd) -> None:
        if self.finished:
            return
        self.finished = True
        elapsed: Final = self.elapsed_ms()
        content_gap: Final = max(self.counts.max_content_gap_ms, elapsed - (self.counts.last_content_ms or 0))
        complete: Final = (
            self.counts.stops == 1
            and self.counts.errors == 0
            and self.counts.starts == 1
            and not self.pending.strip()
            and not self.discarding_frame
            and self.counts.unparsed_frames == 0
            and self.counts.empty_data_frames == 0
            and not self.send_failed
            and (self.stage != "http_send" or self.http_complete)
        )
        slow: Final = content_gap >= _SLOW_CONTENT_MS or self.max_send_ms >= _SLOW_SEND_MS
        if complete and reason == "eof" and (self.mode == "errors" or (self.mode == "slow" and not slow)):
            return
        summary: Final = {
            "stage": self.stage,
            "call_id": self.call_id,
            "upstream_request_id": self.upstream_request_id,
            "end": reason,
            "complete": complete,
            "slow": slow,
            "started_at_unix_ms": self.started_at_unix_ms,
            "started_monotonic_ms": int(self.started * 1000),
            "elapsed_ms": elapsed,
            "pending_bytes": len(self.pending),
            "headers_sent_ms": self.headers_sent_ms,
            "first_send_attempt_ms": self.first_send_attempt_ms,
            "last_send_completed_ms": self.last_send_completed_ms,
            "send_attempts": self.send_attempts,
            "sends_completed": self.sends_completed,
            "send_failed": self.send_failed,
            "send_in_flight_ms": elapsed - self.send_started_ms if self.send_started_ms is not None else None,
            "max_send_ms": self.max_send_ms,
            **asdict(self.counts),
            "max_content_gap_ms": content_gap,
        }
        with suppress(Exception):
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
            self.finish(end)
            await stream.aclose()


def finish_stream_diagnostics(diagnostics: AnthropicStreamDiagnostics | None, reason: StreamEnd) -> None:
    if diagnostics is not None:
        diagnostics.finish(reason)
