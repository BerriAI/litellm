import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.common_request_processing import stream_usage_tracking_updates
from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import CallTypes

_USAGE_FRAME_ENDPOINTS: Final = ("chat/completions", "completions")
_CALLER_ONLY_STREAM_OPTIONS: Final = frozenset({"continuous_usage_stats"})
_SSE_FRAME_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_SSE_DATA_FIELD: Final = b"data:"
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(Mapping[str, object])


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


@dataclass(frozen=True, slots=True)
class _HeldFrame:
    frame: bytes
    earlier: "_HeldFrame | None"


@dataclass(frozen=True, slots=True)
class _RelayState:
    newest_held: _HeldFrame | None = None
    passes_usage_through: bool = False


def _newest_first(held: _HeldFrame | None) -> Iterator[bytes]:
    while held is not None:
        yield held.frame
        held = held.earlier  # rebind-ok: walks the held frames back from the newest


def _held_frames(state: _RelayState) -> tuple[bytes, ...]:
    return tuple(reversed(tuple(_newest_first(state.newest_held))))


def _relays_usage_frames(endpoint: str) -> bool:
    path: Final = endpoint.strip("/")
    return any(path == suffix or path.endswith(f"/{suffix}") for suffix in _USAGE_FRAME_ENDPOINTS)


def _forwarded_stream_options(stream_options: object, strips_usage: bool) -> object:
    if not strips_usage or not isinstance(stream_options, Mapping):
        return stream_options
    caller_options: Final = _JSON_OBJECT_ADAPTER.validate_python(stream_options)
    return {key: value for key, value in caller_options.items() if key not in _CALLER_ONLY_STREAM_OPTIONS}


def stream_usage_relay(
    endpoint: str,
    request_body: Mapping[str, object],
    general_settings: Mapping[str, object],
    supports_stream_options: Callable[[], bool],
) -> StreamUsageRelay:
    if not _relays_usage_frames(endpoint):
        return StreamUsageRelay(body=request_body, strips_usage=False)
    updates: Final = stream_usage_tracking_updates(
        data=request_body,
        general_settings=general_settings,
        route_type=CallTypes.acompletion.value,
        supports_stream_options=supports_stream_options,
    )
    strips_usage: Final = updates.get("_litellm_strip_stream_usage") is True
    stream_options: Final = updates.get("stream_options")
    return StreamUsageRelay(
        body=(
            request_body
            if stream_options is None
            else {**request_body, "stream_options": _forwarded_stream_options(stream_options, strips_usage)}
        ),
        strips_usage=strips_usage,
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
    payload: Final = _JSON_OBJECT_ADAPTER.validate_json(_frame_payload(frame))
    terminator: Final = frame[len(frame.rstrip(b"\r\n")) :]
    restamped: Final = json.dumps(
        {**payload, "system_fingerprint": fingerprint}, separators=(",", ":"), ensure_ascii=False
    )
    return _SSE_DATA_FIELD + b" " + restamped.encode() + terminator


def _awaits_fingerprint(chunk: _RelayedChunk) -> bool:
    return chunk.system_fingerprint is None and any(choice.finish_reason is not None for choice in chunk.choices)


def _relayed_frame(frame: bytes, state: _RelayState) -> tuple[bytes, _RelayState]:
    if state.passes_usage_through:
        return frame, state
    chunk: Final = _relayed_chunk(frame)
    if chunk is None:
        return b"".join((*_held_frames(state), frame)), _RelayState()
    if chunk.usage is not None and chunk.choices:
        return b"".join((*_held_frames(state), frame)), _RelayState(passes_usage_through=True)
    if chunk.usage is not None:
        return (
            b"".join(_with_fingerprint(held, chunk.system_fingerprint) for held in _held_frames(state)),
            _RelayState(),
        )
    if _awaits_fingerprint(chunk):
        return b"", replace(state, newest_held=_HeldFrame(frame, state.newest_held))
    return frame, state


def _sse_frames(complete: bytes) -> tuple[bytes, ...]:
    ends: Final = tuple(match.end() for match in _SSE_FRAME_END.finditer(complete))
    return tuple(complete[start:end] for start, end in zip((0, *ends[:-1]), ends))


async def without_injected_usage_frames(stream: AsyncGenerator[bytes, None]) -> AsyncIterator[bytes]:
    pending = b""  # rebind-ok: carries an unterminated SSE frame across upstream reads
    state = _RelayState()  # rebind-ok: finish frames wait for the fingerprint vLLM moves onto the usage frame
    try:
        async for piece in stream:
            complete, pending = split_complete_sse_frames(pending + piece)
            for frame in _sse_frames(complete):
                outgoing, state = _relayed_frame(frame, state)
                if outgoing:
                    yield outgoing
        if state.newest_held is not None or pending:
            yield b"".join((*_held_frames(state), pending))
    finally:
        await stream.aclose()
