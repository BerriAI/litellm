import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.common_request_processing import stream_usage_tracking_updates
from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames
from litellm.types.llms.base import LiteLLMBaseModel

_STREAM_USAGE_ROUTE_TYPES: Final = (("chat/completions", "acompletion"), ("completions", "atext_completion"))
_SSE_FRAME_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_SSE_DATA_FIELD: Final = b"data:"
_FRAME_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class _RelayedChoice(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    finish_reason: str | None = None


class _RelayedChunk(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    choices: Sequence[_RelayedChoice]
    usage: Mapping[str, object] | None = None
    system_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class StreamUsageRelay:
    body: Mapping[str, object]
    strips_usage: bool


def _stream_usage_route_type(endpoint: str) -> str | None:
    path: Final = endpoint.strip("/")
    return next(
        (
            route_type
            for suffix, route_type in _STREAM_USAGE_ROUTE_TYPES
            if path == suffix or path.endswith(f"/{suffix}")
        ),
        None,
    )


def stream_usage_relay(
    endpoint: str,
    request_body: Mapping[str, object],
    general_settings: Mapping[str, object],
    supports_stream_options: Callable[[], bool],
) -> StreamUsageRelay:
    route_type: Final = _stream_usage_route_type(endpoint)
    if route_type is None:
        return StreamUsageRelay(body=request_body, strips_usage=False)
    updates: Final = stream_usage_tracking_updates(
        data=request_body,
        general_settings=general_settings,
        route_type=route_type,
        supports_stream_options=supports_stream_options,
    )
    stream_options: Final = updates.get("stream_options")
    return StreamUsageRelay(
        body=request_body if stream_options is None else {**request_body, "stream_options": stream_options},
        strips_usage=updates.get("_litellm_strip_stream_usage") is True,
    )


def _frame_payload(frame: bytes) -> bytes:
    return b"\n".join(
        line[len(_SSE_DATA_FIELD) :].strip() for line in frame.splitlines() if line.startswith(_SSE_DATA_FIELD)
    )


def _relayed_chunk(frame: bytes) -> _RelayedChunk | None:
    try:
        return _RelayedChunk.model_validate_json(_frame_payload(frame))
    except ValidationError:
        return None


def _with_fingerprint(frame: bytes, fingerprint: str | None) -> bytes:
    if fingerprint is None:
        return frame
    payload: Final = _FRAME_ADAPTER.validate_json(_frame_payload(frame))
    terminator: Final = frame[len(frame.rstrip(b"\r\n")) :]
    restamped: Final = json.dumps(
        {**payload, "system_fingerprint": fingerprint}, separators=(",", ":"), ensure_ascii=False
    )
    return _SSE_DATA_FIELD + b" " + restamped.encode() + terminator


def _awaits_fingerprint(chunk: _RelayedChunk) -> bool:
    return chunk.system_fingerprint is None and any(choice.finish_reason is not None for choice in chunk.choices)


def _relayed_frame(frame: bytes, held: bytes | None) -> tuple[bytes, bytes | None]:
    chunk: Final = _relayed_chunk(frame)
    if chunk is not None and not chunk.choices and chunk.usage is not None:
        return (b"" if held is None else _with_fingerprint(held, chunk.system_fingerprint)), None
    released: Final = held or b""
    if chunk is not None and _awaits_fingerprint(chunk):
        return released, frame
    return released + frame, None


def _sse_frames(complete: bytes) -> tuple[bytes, ...]:
    ends: Final = tuple(match.end() for match in _SSE_FRAME_END.finditer(complete))
    return tuple(complete[start:end] for start, end in zip((0, *ends[:-1]), ends))


async def without_injected_usage_frames(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    pending = b""  # rebind-ok: carries an unterminated SSE frame across upstream reads
    held: bytes | None = None  # rebind-ok: a finish frame waits for the fingerprint vLLM moves onto the usage frame
    async for piece in stream:
        complete, pending = split_complete_sse_frames(pending + piece)
        for frame in _sse_frames(complete):
            outgoing, held = _relayed_frame(frame, held)
            if outgoing:
                yield outgoing
    if held is not None:
        yield held
    if pending:
        yield pending
