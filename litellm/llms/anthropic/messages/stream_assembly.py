import codecs
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    decoded_json_object,
    is_undecodable_json_object,
    loads_json_object,
    map_finish_reason,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.messages.native.utils import MESSAGE_DELTA_OPTIONAL_KEYS
from litellm.llms.custom_httpx.sse_lines import split_sse_lines
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    TextCompletionResponse,
)

_DECODED_JSON: Final = TypeAdapter(object)


def split_sse_chunk_into_events(chunk: str | bytes) -> list[str]:
    """
    Split a chunk that may contain multiple SSE events into individual events.

    SSE format: "event: type\ndata: {...}\n\n"
    Multiple events in a single chunk are separated by double newlines.

    Args:
        chunk: Raw chunk string that may contain multiple SSE events

    Returns:
        List of individual SSE event strings (each containing "event: X\ndata: {...}")
    """
    # Handle bytes input
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8")

    # Split on double newlines to separate SSE events
    # Filter out empty strings
    events: Final = [event.strip() for event in chunk.split("\n\n") if event.strip()]

    return events


def build_complete_streaming_response(
    all_chunks: Sequence[str | bytes],
    litellm_logging_obj: LiteLLMLoggingObj,
    model: str,
    speed: str | None = None,
) -> ModelResponse | TextCompletionResponse | None:
    """
    Builds complete response from raw Anthropic chunks.

    Fast path: for the dominant case of a pure-text streaming response
    (no tool_use / thinking / non-text content blocks), the long run of
    ``content_block_delta`` text deltas is collapsed into a single
    equivalent SSE event before conversion. ``chunk_parser`` and
    ``stream_chunk_builder`` remain the single source of truth for chunk
    shape, usage math and finish-reason mapping, so the rebuilt response
    (and therefore the logged/billed payload) is identical -- this is
    asserted by a parity test. Anything non-trivial falls back to the
    unchanged legacy reconstruction.

    Per-event Pydantic ``ModelResponseStream`` construction dominated
    event-loop CPU under concurrent streaming; collapsing the homogeneous
    text run removes O(num_output_tokens) of it.
    """
    collapsed: Final = collapse_pure_text_chunks(all_chunks)
    if collapsed is not None:
        return build_complete_streaming_response_legacy(
            all_chunks=collapsed,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            speed=speed,
        )
    return build_complete_streaming_response_legacy(
        all_chunks=all_chunks,
        litellm_logging_obj=litellm_logging_obj,
        model=model,
        speed=speed,
    )


# Anthropic SSE block/delta types that the fast path is NOT allowed to
# collapse -- their presence forces the unchanged legacy path so tool
# calls, thinking, citations, etc. keep byte-identical reconstruction.
_FAST_PATH_DISALLOWED_DELTA_TYPES: Final = frozenset(
    {
        "input_json_delta",
        "thinking_delta",
        "signature_delta",
        "citations_delta",
    }
)


def _joined_text(parts: Sequence[object]) -> str:
    strings: Final = [part for part in parts if isinstance(part, str)]
    if len(strings) == len(parts):
        return "".join(strings)
    first_bad: Final = next(position for position, part in enumerate(parts) if not isinstance(part, str))
    raise TypeError(f"sequence item {first_bad}: expected str instance, {type(parts[first_bad]).__name__} found")


def collapse_pure_text_chunks(
    all_chunks: Sequence[str | bytes],
) -> list[str] | None:
    """
    Return a new chunk list with the contiguous run of text-only
    ``content_block_delta`` events replaced by a single equivalent event,
    or ``None`` if the stream is not a pure single-text-block response
    (in which case the caller uses the legacy path unchanged).

    Only ``message_start`` / ``content_block_start(text)`` /
    ``content_block_delta(text_delta)`` / ``content_block_stop`` /
    ``message_delta`` / ``message_stop`` / ``ping`` events are accepted.
    Any other content-block type or delta type returns ``None``.
    """
    normalized: Final[list[str]] = []
    for raw in all_chunks:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for ev in line.split("\n\n"):
            ev = ev.strip()
            if ev:
                normalized.append(ev)

    text_block_indexes: Final[set[object]] = set()
    out: Final[list[str]] = []
    pending_text: list[object] = []
    pending_index: object = None
    saw_any_text_delta = False

    def flush() -> None:
        nonlocal pending_text, pending_index
        if pending_text:
            merged: Final = {
                "type": "content_block_delta",
                "index": pending_index if pending_index is not None else 0,
                "delta": {"type": "text_delta", "text": _joined_text(pending_text)},
            }
            out.append("data: " + json.dumps(merged))
            pending_text = []
            pending_index = None

    for ev in normalized:
        idx = ev.find("data:")
        if idx == -1:
            # Bare "event: <name>" line. The legacy converter turns this
            # into an empty ModelResponseStream that contributes nothing
            # to stream_chunk_builder. Drop the high-frequency interior
            # markers (content_block_delta / ping); keep every other
            # bare event line verbatim so chunk ordering and the
            # load-bearing chunks[0] (event: message_start) are retained.
            name = ev[len("event:") :].strip() if ev.startswith("event:") else ""
            if name in ("content_block_delta", "ping"):
                continue
            flush()
            out.append(ev)
            continue

        json_str = ev[idx + len("data:") :].strip()
        if not json_str.startswith("{"):
            continue
        data = loads_json_object(json_str)
        if data is None:
            return None

        etype = data.get("type")
        if etype == "content_block_start":
            block = decoded_json_object(data.get("content_block") or {})
            if block.get("type") != "text":
                return None
            text_block_indexes.add(data.get("index"))
            flush()
            out.append(ev)
        elif etype == "content_block_delta":
            delta = decoded_json_object(data.get("delta") or {})
            dtype = delta.get("type")
            if dtype in _FAST_PATH_DISALLOWED_DELTA_TYPES:
                return None
            if dtype != "text_delta":
                return None
            cur_index = data.get("index")
            if cur_index not in text_block_indexes:
                return None
            # Defensive: Anthropic sends blocks strictly sequentially
            # (start/deltas/stop, then next block), so pending_text from
            # block N must be flushed by content_block_stop before block
            # N+1's deltas arrive. If we ever see a delta whose index
            # disagrees with the current pending buffer, the stream is
            # interleaved -- fall back to legacy rather than risk merging
            # text from different blocks under a single index.
            if pending_text and pending_index is not None and cur_index != pending_index:
                return None
            saw_any_text_delta = True
            pending_index = cur_index
            pending_text.append(delta.get("text") or "")
        elif etype == "ping":
            # Interior no-op; legacy maps it to an empty chunk.
            continue
        else:
            # message_start / content_block_stop / message_delta /
            # message_stop / error: pass through unchanged.
            flush()
            out.append(ev)

    flush()

    if not saw_any_text_delta:
        return None
    return out


def build_complete_streaming_response_legacy(
    all_chunks: Sequence[str | bytes],
    litellm_logging_obj: LiteLLMLoggingObj,
    model: str,
    speed: str | None = None,
) -> ModelResponse | TextCompletionResponse | None:
    """
    Original reconstruction: convert every SSE event to a generic chunk
    and assemble via stream_chunk_builder. Kept verbatim as the fallback
    / source of truth for the fast path's parity test.

    - Splits multi-event chunks into individual SSE events
    - Converts str chunks to generic chunks
    - Converts generic chunks to litellm chunks (OpenAI format)
    - Builds complete response from litellm chunks
    """
    verbose_proxy_logger.debug("Building complete streaming response from %d chunks", len(all_chunks))
    anthropic_model_response_iterator: Final = AnthropicModelResponseIterator(
        streaming_response=None,
        sync_stream=False,
        speed=speed,
    )
    all_openai_chunks: Final = []

    # Process each chunk - a chunk may contain multiple SSE events
    for _chunk_str in all_chunks:
        # Split chunk into individual SSE events
        individual_events = split_sse_chunk_into_events(_chunk_str)

        # Process each individual event
        for event_str in individual_events:
            try:
                # Skip OpenAI-style [DONE] sentinels some Anthropic-compatible
                # providers emit. Match the whole SSE line so a valid chunk whose
                # text payload happens to contain "[DONE]" is not dropped.
                if any(line.strip() == "data: [DONE]" for line in event_str.split("\n")):
                    continue
                transformed_openai_chunk = anthropic_model_response_iterator.convert_str_chunk_to_generic_chunk(
                    chunk=event_str
                )
                if transformed_openai_chunk is not None:
                    all_openai_chunks.append(transformed_openai_chunk)

            except (StopIteration, StopAsyncIteration):
                break
            except json.JSONDecodeError:
                # Some upstreams emit non-JSON SSE lines; skip them so the
                # logging pipeline is not broken by a single bad frame.
                verbose_proxy_logger.debug(
                    "Skipping non-JSON SSE event: %s",
                    event_str[:200],
                )
                continue

    complete_streaming_response: Final = litellm.stream_chunk_builder(
        chunks=all_openai_chunks,
        logging_obj=litellm_logging_obj,
    )
    verbose_proxy_logger.debug("Complete streaming response built: %s", complete_streaming_response)
    return complete_streaming_response


def _sse_data_payload(event_str: str) -> str | None:
    for line in split_sse_lines(event_str):
        stripped = line.strip()
        if stripped.startswith("data:"):
            return stripped[len("data:") :].strip()
    return None


def extract_sse_data(event_str: str) -> dict | None:
    """Parse the JSON object from the ``data:`` line of an Anthropic SSE event."""
    payload: Final = _sse_data_payload(event_str)
    decoded: Final = None if payload is None else loads_json_object(payload)
    return decoded if isinstance(decoded, dict) else None


def build_usage_only_response_from_chunks(
    all_chunks: Sequence[str | bytes],
    model: str,
    speed: str | None = None,
) -> ModelResponse | None:
    """
    Build a usage-bearing ModelResponse from Anthropic SSE token-usage events, for
    cost tracking when stream_chunk_builder cannot reassemble the stream.

    Anthropic emits usage in ``message_start`` (uncached input + cache tokens, and an
    initial output_tokens) and the final ``message_delta`` (cumulative output_tokens)
    regardless of the content/tool shape, so cost is recoverable even when full
    content assembly fails. Returns ``None`` if no usage event is found.
    """
    input_tokens = 0
    cache_read = 0
    cache_creation = 0
    cache_creation_5m: int | None = None
    cache_creation_1h: int | None = None
    output_tokens = 0
    web_search_requests: int | None = None
    tool_search_requests: int | None = None
    inference_geo: str | None = None
    speed_from_stream: str | None = None
    stop_reason: str | None = None
    found_usage = False
    resolved_model = model
    for _chunk_str in all_chunks:
        for event_str in split_sse_chunk_into_events(_chunk_str):
            data = extract_sse_data(event_str)
            if not data:
                continue
            event_type = data.get("type")
            if event_type == "message_start":
                message = data.get("message") or {}
                if not resolved_model or resolved_model == "unknown":
                    resolved_model = message.get("model") or resolved_model
                usage = message.get("usage") or {}
                input_tokens = usage.get("input_tokens") or input_tokens
                cache_read = usage.get("cache_read_input_tokens") or cache_read
                cache_creation = usage.get("cache_creation_input_tokens") or cache_creation
                _cc = usage.get("cache_creation")
                if isinstance(_cc, dict):
                    cache_creation_5m = _cc.get("ephemeral_5m_input_tokens")
                    cache_creation_1h = _cc.get("ephemeral_1h_input_tokens")
                if usage.get("inference_geo") is not None:
                    inference_geo = usage.get("inference_geo")
                if isinstance(usage.get("speed"), str):
                    speed_from_stream = usage.get("speed")
                if usage.get("output_tokens") is not None:
                    output_tokens = usage.get("output_tokens")
                found_usage = True
            elif event_type == "message_delta":
                _delta_stop = (data.get("delta") or {}).get("stop_reason")
                if _delta_stop:
                    stop_reason = _delta_stop
                usage = data.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    output_tokens = usage.get("output_tokens")
                _stu = usage.get("server_tool_use")
                if isinstance(_stu, dict):
                    if _stu.get("web_search_requests") is not None:
                        web_search_requests = _stu.get("web_search_requests")
                    if _stu.get("tool_search_requests") is not None:
                        tool_search_requests = _stu.get("tool_search_requests")
                if usage.get("cache_read_input_tokens") is not None:
                    cache_read = usage.get("cache_read_input_tokens")
                if usage.get("inference_geo") is not None:
                    inference_geo = usage.get("inference_geo")
                if isinstance(usage.get("speed"), str):
                    speed_from_stream = usage.get("speed")
                found_usage = True
    if not found_usage:
        return None
    # If only the 5m/1h split was provided, derive the cache_creation total from it.
    if not cache_creation and (cache_creation_5m or cache_creation_1h):
        cache_creation = (cache_creation_5m or 0) + (cache_creation_1h or 0)
    # build usage via the same AnthropicConfig.calculate_usage path the success
    # cases use, so prompt_tokens are cache-inclusive and cache / server_tool_use /
    # inference_geo tokens are priced instead of left at $0
    usage_object: Final[dict] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read:
        usage_object["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage_object["cache_creation_input_tokens"] = cache_creation
    if cache_creation_5m is not None or cache_creation_1h is not None:
        usage_object["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_creation_5m or 0,
            "ephemeral_1h_input_tokens": cache_creation_1h or 0,
        }
    if web_search_requests is not None or tool_search_requests is not None:
        _server_tool_use: Final[dict] = {}
        if web_search_requests is not None:
            _server_tool_use["web_search_requests"] = web_search_requests
        if tool_search_requests is not None:
            _server_tool_use["tool_search_requests"] = tool_search_requests
        usage_object["server_tool_use"] = _server_tool_use
    if inference_geo is not None:
        usage_object["inference_geo"] = inference_geo
    if speed_from_stream is not None:
        usage_object["speed"] = speed_from_stream
    usage_obj: Final = AnthropicConfig().calculate_usage(usage_object=usage_object, reasoning_content=None, speed=speed)
    return ModelResponse(
        model=resolved_model,
        choices=[
            Choices(
                finish_reason=(map_finish_reason(stop_reason) if stop_reason else "stop"),
                index=0,
                message=Message(role="assistant", content=""),
            )
        ],
        usage=usage_obj,
    )


_ANTHROPIC_EVENT_TYPES: Final = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    }
)


_ERROR_PAYLOAD_KEYS: Final = frozenset({"type", "error", "request_id"})


def _is_anthropic_event_type(event_type: object) -> bool:
    return isinstance(event_type, str) and event_type in _ANTHROPIC_EVENT_TYPES


def is_raw_sse_stream(all_chunks: Sequence[object]) -> bool:
    return any(isinstance(chunk, (str, bytes)) for chunk in all_chunks)


def _joined_sse_bytes(all_chunks: Sequence[object]) -> bytes:
    return b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        for chunk in all_chunks
        if isinstance(chunk, (str, bytes))
    )


def _joined_sse_stream(all_chunks: Sequence[object]) -> str | None:
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(_joined_sse_bytes(all_chunks), final=False)
    except UnicodeDecodeError:
        return None


_JSON_ESCAPED_LINE_BREAKS: Final = MappingProxyType(
    str.maketrans({"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"})
)


def _sse_frames(sse_stream: str) -> Sequence[str]:
    return "\n".join(split_sse_lines(sse_stream)).split("\n\n")


def _frames_of_the_widest_reader(sse_stream: str) -> Sequence[str]:
    """The stream's frames as httpx's ``str.splitlines`` reads them, past a form feed or U+2028 in any line."""
    return "\n".join(sse_stream.splitlines()).split("\n\n")


def _sse_field(line: str) -> tuple[str, str]:
    name, _, value = line.partition(":")
    return name, value.removeprefix(" ")


def _sse_fields(frame: str) -> tuple[tuple[str, str], ...]:
    """The frame's fields as a client reads them: a name matches exactly and a comment is no field."""
    return tuple(_sse_field(line) for line in frame.split("\n") if line and not line.startswith(":"))


def _carries_no_data(fields: Sequence[tuple[str, str]]) -> bool:
    match fields:
        case () | (("event", _),) | (("data", "[DONE]"),):
            return True
        case _:
            return False


_EVENT_NAMES_A_CLIENT_APPLIES: Final = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)

_BYTE_ORDER_MARK: Final = "\ufeff"


def _is_applied_by_name(value: object) -> bool:
    return isinstance(value, str) and value in _EVENT_NAMES_A_CLIENT_APPLIES


def _client_frame(frame: str) -> tuple[str, str] | None:
    fields: Final = _sse_fields(frame)
    data_lines: Final = tuple(value for name, value in fields if name == "data")
    if not data_lines:
        return None
    names: Final = tuple(value for name, value in fields if name == "event")
    return names[-1] if names else "", "\n".join(data_lines)


def _client_frames(frames: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The ``event`` name and joined ``data`` of every frame a client dispatches, which is every frame with data."""
    return tuple(client_frame for frame in frames if (client_frame := _client_frame(frame)) is not None)


_NamedData: TypeAlias = tuple[str, Mapping[str, object] | None]


def _named_data(client_frames: Sequence[tuple[str, str]]) -> tuple[_NamedData, ...]:
    return tuple((name, loads_json_object(data)) for name, data in client_frames)


def _client_event(name: str, data: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if data is None:
        return None
    event: Final = {**data, "type": name} if "type" not in data and _is_applied_by_name(name) else data
    return event if _is_anthropic_event_type(event.get("type")) else None


def _client_events(named_data: Sequence[_NamedData]) -> tuple[Mapping[str, object], ...]:
    return tuple(event for name, data in named_data if (event := _client_event(name, data)) is not None)


def _stream_client_frames(sse_stream: str) -> tuple[tuple[str, str], ...]:
    return _client_frames(_sse_frames(sse_stream.removeprefix(_BYTE_ORDER_MARK)))


def _parsed_sse_events(sse_stream: str) -> tuple[Mapping[str, object], ...]:
    """Every Anthropic event a client reads in the stream, whatever line breaks, comments or data lines it uses.

    An event without a data ``type`` takes its SSE ``event:`` name, as the Anthropic SDK reads it.
    """
    return _client_events(_named_data(_stream_client_frames(sse_stream)))


@dataclass(frozen=True, slots=True)
class _UnappliedFrame:
    text: str


_StreamItem: TypeAlias = Mapping[str, object] | _UnappliedFrame


def _is_one_line(text: str) -> bool:
    return len(text.splitlines()) <= 1


def _stream_item(name: str, data: str) -> _StreamItem | None:
    """The frame's event, else the frame itself when no client applies it, like an unknown event or ``[DONE]``."""
    event: Final = _client_event(name, loads_json_object(data))
    if event is not None:
        return event
    if name in _ANTHROPIC_EVENT_TYPES or not _is_one_line(name + data):
        return None
    return _UnappliedFrame(f"event: {name}\ndata: {data}\n\n" if name else f"data: {data}\n\n")


def _stream_items(sse_stream: str) -> tuple[_StreamItem, ...]:
    return tuple(
        item for name, data in _stream_client_frames(sse_stream) if (item := _stream_item(name, data)) is not None
    )


def _anthropic_message_start(events: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    return next(
        (
            message
            for event_data in events
            if event_data.get("type") == "message_start"
            and (message := as_str_mapping(event_data.get("message"))) is not None
        ),
        None,
    )


def anthropic_message_delta_fields(all_chunks: Sequence[object]) -> Mapping[str, object]:
    """The ``stop_details``, ``safeguard_results`` and ``container`` of the stream's ``message_delta``.

    The assembled OpenAI shape has no slot for them, and a client that asked to stream reads them
    from that event and nowhere else (a Claude Code auto-mode session takes its classifier verdict
    from ``safeguard_results``), so a caller re-emitting a rewritten response merges them back in.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return MappingProxyType({})
    message_delta: Final = next(
        (event for event in reversed(_parsed_sse_events(sse_stream)) if event.get("type") == "message_delta"), None
    )
    delta: Final = as_str_mapping(message_delta.get("delta")) if message_delta is not None else None
    if delta is None:
        return MappingProxyType({})
    return MappingProxyType(
        {key: value for key in MESSAGE_DELTA_OPTIONAL_KEYS if (value := delta.get(key)) is not None}
    )


def is_anthropic_sse_stream(all_chunks: Sequence[object]) -> bool:
    """Whether raw SSE frames are Anthropic Messages events.

    ``is_raw_sse_stream`` only says the chunks are unparsed bytes, and ``/v1/messages`` is not the
    only endpoint that streams those: the Google ``:streamGenerateContent`` route marks its own
    stream raw too. Reading its frames as Anthropic ones would refuse the response in a wire format
    its client cannot parse, so the surface is decided on the event types actually present.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    return any(_is_anthropic_event_type(event.get("type")) for event in _parsed_sse_events(sse_stream))


def _is_json_no_reader_can_take(payload: str) -> bool:
    try:
        value: Final = _DECODED_JSON.validate_python(json.loads(payload))
    except json.JSONDecodeError:
        return False
    except (ValueError, RecursionError):
        return True
    if value is None:
        return False
    event: Final = as_str_mapping(value)
    return event is None or isinstance(event.get("type"), (list, dict))


def has_unreadable_json_event(all_chunks: Sequence[object]) -> bool:
    """Whether a frame's data is JSON that no reader of these streams can take as an event."""
    sse_stream: Final = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(
        _joined_sse_bytes(all_chunks), final=False
    )
    return any(_is_json_no_reader_can_take(_sse_data_payload(event) or "") for event in _sse_frames(sse_stream))


def _has_undecodable_object(client_frames: Sequence[tuple[str, str]]) -> bool:
    return any(is_undecodable_json_object(data) for _, data in client_frames)


def anthropic_sse_event_type(event: str) -> object:
    return next((event_data.get("type") for event_data in _parsed_sse_events(event)), None)


def anthropic_sse_event_types(all_chunks: Sequence[object]) -> frozenset[str]:
    """The string ``type`` of every event the stream sent."""
    event_types: Final = (event.get("type") for event in _parsed_sse_events(_joined_sse_stream(all_chunks) or ""))
    return frozenset(event_type for event_type in event_types if isinstance(event_type, str))


def _is_named_error_event(event: str) -> bool:
    names: Final = tuple(value for name, value in _sse_fields(event) if name == "event")
    return names[-1:] == ("error",)


def _is_error_event(event: str) -> bool:
    event_type: Final = anthropic_sse_event_type(event)
    return event_type == "error" or (_is_named_error_event(event) and not _is_anthropic_event_type(event_type))


def _frames_a_client_dispatches(sse_stream: str) -> Sequence[str]:
    return "\n".join((*split_sse_lines(sse_stream), "")).split("\n\n")[:-1]


def _error_events(sse_stream: str) -> tuple[str, ...]:
    return tuple(event for event in _frames_a_client_dispatches(sse_stream) if _is_error_event(event))


def _plain_error_event(event: str) -> str:
    payload: Final = _sse_data_payload(event) or ""
    one_line: Final = (
        payload.translate(_JSON_ESCAPED_LINE_BREAKS) if loads_json_object(payload) is not None else payload
    )
    return "\n".join(("event: error", *(f"data: {line}" for line in one_line.splitlines()[:1])))


def anthropic_sse_error_events(all_chunks: Sequence[object]) -> tuple[str, ...]:
    """The ``error`` events of the stream, each as an ``event:`` line and the ``data:`` line assembly read.

    A stream whose upstream fails after it started ends with an ``error`` event in place of
    ``message_stop``. That frame is the outcome the client is owed, not content, so assembly through
    the upstream error leaves it out and a caller re-emitting a rewritten response puts it back,
    without any other line of the frame, since that is content no guardrail read.
    A frame counts by its data ``type``, or by its SSE ``event:`` name unless its data is another
    Anthropic event, since a client SDK raises on the name alone and an upstream that is not
    Anthropic may send anything as the data.
    """
    return tuple(_plain_error_event(event) for event in _error_events(_joined_sse_stream(all_chunks) or ""))


def anthropic_sse_events_before_upstream_error(all_chunks: Sequence[str | bytes]) -> Sequence[str | bytes]:
    """The stream without its upstream ``error`` events, so what arrived before the failure still assembles."""
    events: Final = tuple(chain.from_iterable(map(split_sse_chunk_into_events, all_chunks)))
    received: Final = tuple(event for event in events if "error" not in event or not _is_error_event(event))
    return all_chunks if len(received) == len(events) else received


@dataclass(frozen=True, slots=True)
class AnthropicSseFrame:
    """A frame of the stream, with the Anthropic event a client reads in it and the data of an error one raises on."""

    raw: str
    event: Mapping[str, object]
    error_data: str | None


def _sse_frame(raw: str) -> AnthropicSseFrame:
    name, data = _client_frame(raw) or ("", "")
    return AnthropicSseFrame(
        raw=raw,
        event=_client_event(name, loads_json_object(data)) or MappingProxyType({}),
        error_data=data if _is_error_event(raw) or _is_named_error_event(raw) else None,
    )


def anthropic_sse_frames(all_chunks: Sequence[object]) -> tuple[AnthropicSseFrame, ...]:
    return tuple(map(_sse_frame, _frames_a_client_dispatches(_joined_sse_stream(all_chunks) or "")))


def _error_data_lines(data: str) -> tuple[str, ...]:
    if loads_json_object(data) is not None:
        return (data.replace("\n", " ").translate(_JSON_ESCAPED_LINE_BREAKS),)
    return tuple(data.split("\n")) if data else ()


def anthropic_sse_error_event_of(data: str) -> str:
    return "\n".join(("event: error", *(f"data: {line}" for line in _error_data_lines(data))))


def anthropic_sse_stopped_block_index(event: str) -> object:
    return next(
        (
            event_data.get("index")
            for event_data in _parsed_sse_events(event)
            if event_data.get("type") == "content_block_stop"
        ),
        None,
    )


def _block_index(event_data: Mapping[str, object]) -> int | None:
    index: Final = event_data.get("index")
    return index if isinstance(index, int) else None


def _started_blocks(events: Sequence[Mapping[str, object]]) -> Mapping[int, Mapping[str, object]]:
    starts: Final = (event for event in events if event.get("type") == "content_block_start")
    return MappingProxyType(
        {
            index: block
            for event in starts
            if (index := _block_index(event)) is not None
            and (block := as_str_mapping(event.get("content_block"))) is not None
        }
    )


def anthropic_sse_started_blocks(all_chunks: Sequence[object]) -> Mapping[int, Mapping[str, object]]:
    """The ``content_block`` each ``content_block_start`` of the stream opened, by block index."""
    return _started_blocks(_parsed_sse_events(_joined_sse_stream(all_chunks) or ""))


def _unfinished_blocks(events: Sequence[Mapping[str, object]]) -> Mapping[int, Mapping[str, object]]:
    stopped: Final = frozenset(_block_index(event) for event in events if event.get("type") == "content_block_stop")
    return MappingProxyType({index: block for index, block in _started_blocks(events).items() if index not in stopped})


def anthropic_sse_unfinished_blocks(all_chunks: Sequence[object]) -> Mapping[int, Mapping[str, object]]:
    """The blocks the stream opened and never stopped."""
    return _unfinished_blocks(_parsed_sse_events(_joined_sse_stream(all_chunks) or ""))


def _content_events_before_upstream_error(
    sse_stream: str, events: Sequence[Mapping[str, object]]
) -> Sequence[Mapping[str, object]]:
    """The stream's events minus its upstream ``error`` and every cut-off block assembly cannot read.

    A cut-off text block is still text the client was sent, so it stays. A cut-off tool call's
    input is a JSON prefix, and a cut-off thinking block has no signature, so neither is a block the
    client can act on, and both are left out.
    """
    if not _error_events(sse_stream):
        return events
    cut_off: Final = frozenset(
        index for index, block in _unfinished_blocks(events).items() if block.get("type") != "text"
    )
    return tuple(event for event in events if event.get("type") != "error" and _block_index(event) not in cut_off)


def _streamed_input_index(event: Mapping[str, object]) -> int | None:
    delta: Final = as_str_mapping(event.get("delta")) if event.get("type") == "content_block_delta" else None
    if delta is None or delta.get("type") != "input_json_delta" or not delta.get("partial_json"):
        return None
    return _block_index(event)


def _started_tool_use(event: Mapping[str, object]) -> Mapping[str, object] | None:
    block: Final = as_str_mapping(event.get("content_block")) if event.get("type") == "content_block_start" else None
    return block if block is not None and block.get("type") == "tool_use" else None


def _with_tool_input_as_delta(
    event: Mapping[str, object], streamed_input_indexes: frozenset[int]
) -> tuple[Mapping[str, object], ...]:
    """A tool call's start event, with the input it carries moved into the delta assembly reads input from.

    A client takes a ``tool_use`` start's ``input`` as the call's input unless input deltas replace it.
    """
    block: Final = _started_tool_use(event)
    tool_input: Final = as_str_mapping(block.get("input")) if block is not None else None
    if block is None or not tool_input:
        return (event,)
    start: Final = {**event, "content_block": {**block, "input": dict[str, object]()}}
    if _block_index(event) in streamed_input_indexes:
        return (start,)
    input_delta: Final = {"type": "input_json_delta", "partial_json": json.dumps(tool_input)}
    return start, {"type": "content_block_delta", "index": event.get("index"), "delta": input_delta}


def _streamed_input_indexes(events: Sequence[Mapping[str, object]]) -> frozenset[int]:
    return frozenset(index for event in events if (index := _streamed_input_index(event)) is not None)


_BLOCK_EVENT_TYPES: Final = frozenset({"content_block_start", "content_block_delta", "content_block_stop"})


def _positions_by_block(events: Sequence[Mapping[str, object]], event_type: str) -> Mapping[int, tuple[int, ...]]:
    indexed: Final = sorted(
        (index, position)
        for position, event in enumerate(events)
        if event.get("type") == event_type and (index := _block_index(event)) is not None
    )
    return MappingProxyType(
        {
            index: tuple(position for _, position in positions)
            for index, positions in groupby(indexed, key=lambda indexed_position: indexed_position[0])
        }
    )


_DELTA_FIELD_BY_TYPE: Final = MappingProxyType(
    {
        "text_delta": "text",
        "citations_delta": "citation",
        "input_json_delta": "partial_json",
        "thinking_delta": "thinking",
        "signature_delta": "signature",
    }
)

_DELTA_FIELDS: Final = frozenset(_DELTA_FIELD_BY_TYPE.values())


def _delta_as_its_type_reads(event: Mapping[str, object]) -> Mapping[str, object]:
    """A block delta with only the field its ``type`` names, which is the one field an Anthropic SDK applies."""
    delta: Final = as_str_mapping(event.get("delta")) if event.get("type") == "content_block_delta" else None
    if delta is None:
        return event
    delta_type: Final = delta.get("type")
    own_field: Final = _DELTA_FIELD_BY_TYPE.get(delta_type) if isinstance(delta_type, str) else None
    return {
        **event,
        "delta": {key: value for key, value in delta.items() if key == own_field or key not in _DELTA_FIELDS},
    }


_TYPESCRIPT_SDK_INPUT_BUFFER: Final = "__json_buf"


def _carries_the_typescript_sdk_input_buffer(event: Mapping[str, object]) -> bool:
    block: Final = as_str_mapping(event.get("content_block"))
    return block is not None and _TYPESCRIPT_SDK_INPUT_BUFFER in block


def _without_the_typescript_sdk_input_buffer(event: Mapping[str, object]) -> Mapping[str, object]:
    """A block start without the ``__json_buf`` the TypeScript SDK puts in front of the block's input deltas."""
    block: Final = as_str_mapping(event.get("content_block")) if event.get("type") == "content_block_start" else None
    if block is None or _TYPESCRIPT_SDK_INPUT_BUFFER not in block:
        return event
    return {
        **event,
        "content_block": {key: value for key, value in block.items() if key != _TYPESCRIPT_SDK_INPUT_BUFFER},
    }


_BLOCK_CONTENT_EVENT_TYPES: Final = frozenset({"content_block_delta", "content_block_stop"})


def _event_type(item: _StreamItem) -> object:
    return item.get("type") if isinstance(item, Mapping) else None


def _block_order_key(starts: Sequence[int], position: int, item: _StreamItem) -> tuple[int, bool, int]:
    """Sorts a delta or stop right after the start of the block at its ``index``, as both Anthropic SDKs apply it."""
    index: Final = (
        item.get("index") if isinstance(item, Mapping) and _event_type(item) in _BLOCK_CONTENT_EVENT_TYPES else None
    )
    block_start: Final = starts[index] if type(index) is int and 0 <= index < len(starts) else position
    return min(block_start, position), _event_type(item) == "content_block_stop", position


def _in_block_order(items: Sequence[_StreamItem]) -> tuple[_StreamItem, ...]:
    starts: Final = tuple(position for position, item in enumerate(items) if _event_type(item) == "content_block_start")
    ordered: Final = sorted(enumerate(items), key=lambda positioned: _block_order_key(starts, *positioned))
    return tuple(item for _, item in ordered)


def _with_every_block_stopped(events: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """The events plus a stop for each unstopped block, since assembly records a hosted call only at its stop."""
    return (*events, *({"type": "content_block_stop", "index": index} for index in _unfinished_blocks(events)))


def _as_the_sdks_apply(item: _StreamItem) -> _StreamItem:
    if isinstance(item, _UnappliedFrame):
        return item
    return _without_the_typescript_sdk_input_buffer(_delta_as_its_type_reads(item))


def _with_its_input_as_delta(item: _StreamItem, streamed_input_indexes: frozenset[int]) -> tuple[_StreamItem, ...]:
    return (item,) if isinstance(item, _UnappliedFrame) else _with_tool_input_as_delta(item, streamed_input_indexes)


def _sse_frame_text(item: _StreamItem) -> str:
    if isinstance(item, _UnappliedFrame):
        return item.text
    return f"event: {item.get('type')}\ndata: {json.dumps(item)}\n\n"


def _canonical_sse_frames(items: Sequence[_StreamItem]) -> tuple[str, ...]:
    sdk_items: Final = tuple(_as_the_sdks_apply(item) for item in _in_block_order(items))
    streamed_input_indexes: Final = _streamed_input_indexes(
        tuple(item for item in sdk_items if isinstance(item, Mapping))
    )
    canonical_items: Final = chain.from_iterable(
        _with_its_input_as_delta(item, streamed_input_indexes) for item in sdk_items
    )
    return tuple(_sse_frame_text(item) for item in canonical_items)


def anthropic_sse_frames_as_read(all_chunks: Sequence[object]) -> tuple[bytes, ...]:
    """The stream as assembly reads it, one ``event:`` and one ``data:`` line a frame, each block's events together.

    Every field the upstream sent stays except a ``__json_buf`` the TypeScript SDK would read as tool input, and a
    frame no client applies, like an unknown event or ``[DONE]``, stays as sent.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return ()
    return tuple(frame.encode() for frame in _canonical_sse_frames(_stream_items(sse_stream)))


def assemble_anthropic_sse_stream(
    all_chunks: Sequence[object], *, restore_identity: bool = False, through_upstream_error: bool = False
) -> ModelResponse | None:
    """Assemble raw Anthropic SSE frames into a ModelResponse.

    ``restore_identity`` stamps the upstream message id and model onto the result, which the
    assembler does not carry through. It is off by default so callers that re-emit the assembled
    response keep the wire shape they had before this helper was shared. The writes land on a
    freshly built object that is unreachable from caller state until returned.

    A stream whose upstream failed after it started ends with an ``error`` event in place of
    ``message_stop`` and assembles to None, unless ``through_upstream_error`` asks for the content
    the client was sent before the failure. That is for a guardrail that scans the stream and then
    passes the upstream error on; a caller recording what was served keeps None.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    client_frames: Final = _stream_client_frames(sse_stream)
    if _has_undecodable_object(client_frames):
        return None
    events: Final = _client_events(_named_data(client_frames))
    message_start: Final = _anthropic_message_start(events)
    if message_start is None:
        return None
    model: Final = message_start.get("model") if restore_identity else None
    content_events: Final = (
        _content_events_before_upstream_error(sse_stream, events) if through_upstream_error else events
    )
    try:
        assembled: Final = build_complete_streaming_response(
            all_chunks=_canonical_sse_frames(_with_every_block_stopped(content_events)),
            litellm_logging_obj=None,  # pyright: ignore[reportArgumentType]  # only forwarded to stream_chunk_builder, which accepts None
            model=model if isinstance(model, str) else "",
        )
    except Exception:  # noqa: BLE001  # stream_chunk_builder re-raises every assembly failure as litellm.APIError
        return None
    if not isinstance(assembled, ModelResponse):
        return None
    if not restore_identity:
        return assembled
    message_id: Final = message_start.get("id")
    if isinstance(message_id, str):
        assembled.id = message_id
    if isinstance(model, str) and model:
        assembled.model = model
    return assembled


def _input_delta_index(event: Mapping[str, object]) -> int | None:
    delta: Final = as_str_mapping(event.get("delta")) if event.get("type") == "content_block_delta" else None
    return _block_index(event) if delta is not None and delta.get("type") == "input_json_delta" else None


def _replaces_its_start_input(event: Mapping[str, object], input_delta_indexes: frozenset[int]) -> bool:
    block: Final = as_str_mapping(event.get("content_block")) if event.get("type") == "content_block_start" else None
    tool_input: Final = as_str_mapping(block.get("input")) if block is not None else None
    return bool(tool_input) and _block_index(event) in input_delta_indexes


def anthropic_sse_assembles_every_tool_input(all_chunks: Sequence[object]) -> bool:
    """Whether assembly reads the input of every client and server tool call, as it does in a stream that ends well.

    It leaves out a start's input that deltas replace, and reads a server tool call only once it stops.
    """
    events: Final = _parsed_sse_events(_joined_sse_stream(all_chunks) or "")
    input_delta_indexes: Final = frozenset(
        index for event in events if (index := _input_delta_index(event)) is not None
    )
    return not any(_replaces_its_start_input(event, input_delta_indexes) for event in events) and not any(
        block.get("type") == "server_tool_use" for block in _unfinished_blocks(events).values()
    )


def _names_its_own_type(name: str, data: Mapping[str, object] | None) -> bool:
    data_type: Final = data.get("type") if data is not None else None
    return name in (data_type, "error") or not (_is_applied_by_name(name) or _is_applied_by_name(data_type))


def _carries_text(event: Mapping[str, object]) -> bool:
    content: Final = as_str_mapping(event.get("content_block") or event.get("delta"))
    return content is not None and bool(content.get("text"))


def _is_read_by_block(events: Sequence[Mapping[str, object]]) -> bool:
    """Whether the blocks start at 0, 1, 2 and so on, take their deltas while open, and send their text in that order.

    Blocks may overlap, as when an upstream streams several tool calls at once.
    """
    block_events: Final = tuple(event for event in events if event.get("type") in _BLOCK_EVENT_TYPES)
    indexes: Final = tuple(index for event in block_events if type(index := event.get("index")) is int)
    if len(indexes) != len(block_events):
        return False
    started: Final = tuple(event.get("index") for event in block_events if event.get("type") == "content_block_start")
    if started != tuple(range(len(started))):
        return False
    starts: Final = _positions_by_block(block_events, "content_block_start")
    stops: Final = _positions_by_block(block_events, "content_block_stop")
    never: Final = (len(block_events),)
    text_indexes: Final = tuple(
        index for index, event in zip(indexes, block_events, strict=True) if _carries_text(event)
    )
    return text_indexes == tuple(sorted(text_indexes)) and all(
        starts.get(index, never)[0] < position <= stops.get(index, never)[0]
        for position, (index, event) in enumerate(zip(indexes, block_events, strict=True))
        if event.get("type") != "content_block_start"
    )


def _opens_with_an_empty_message(events: Sequence[Mapping[str, object]]) -> bool:
    message_starts: Final = tuple(
        position for position, event in enumerate(events) if event.get("type") == "message_start"
    )
    message: Final = as_str_mapping(events[0].get("message")) if message_starts == (0,) else None
    return message is not None and not message.get("content")


_DELTA_FIELDS_ASSEMBLY_READS_FIRST: Final = MappingProxyType(
    {
        "input_json_delta": frozenset({"text"}),
        "citations_delta": frozenset({"text", "partial_json"}),
        "thinking_delta": frozenset({"text", "partial_json", "citation"}),
        "signature_delta": frozenset({"text", "partial_json", "citation"}),
        "compaction_delta": frozenset({"text", "partial_json", "citation", "thinking", "signature"}),
    }
)


def _assembly_reads_the_field_the_delta_type_names(event: Mapping[str, object]) -> bool:
    delta: Final = as_str_mapping(event.get("delta"))
    delta_type: Final = delta.get("type") if delta is not None else None
    if delta is None or not isinstance(delta_type, str):
        return True
    return delta.keys().isdisjoint(_DELTA_FIELDS_ASSEMBLY_READS_FIRST.get(delta_type, frozenset()))


def _is_plain(line: str) -> bool:
    return not line[:1].isspace() and ("data:" not in line or line.startswith("data:"))


def _has_only_plain_lines(sse_stream: str) -> bool:
    """No bare CR, which an LF-only reader keeps inside a line, and no line a stripping or substring reader reads."""
    return sse_stream.count("\r") == sse_stream.count("\r\n") and all(map(_is_plain, sse_stream.splitlines()))


def _ends_at_its_first_message_stop(events: Sequence[Mapping[str, object]]) -> bool:
    stops: Final = tuple(position for position, event in enumerate(events) if event.get("type") == "message_stop")
    return stops[:1] in ((), (len(events) - 1,))


def _block_events_read_alike(event: Mapping[str, object]) -> bool:
    match event.get("type"):
        case "content_block_start":
            return not _carries_the_typescript_sdk_input_buffer(event)
        case "content_block_delta":
            return _assembly_reads_the_field_the_delta_type_names(event)
        case _:
            return True


def _is_json_object(data: str) -> bool:
    return loads_json_object(data) is not None or is_undecodable_json_object(data)


def _carries_json(client_frame: tuple[str, str]) -> bool:
    _, data = client_frame
    return _is_json_object(data)


def _no_data_line_of_a_frame_is_json_alone(client_frames: tuple[tuple[str, str], ...]) -> bool:
    multi_line_data: Final = (data for _, data in client_frames if "\n" in data)
    return not any(map(_is_json_object, chain.from_iterable(map(str.splitlines, multi_line_data))))


def _the_widest_reader_dispatches_no_other_json(sse_stream: str, client_frames: tuple[tuple[str, str], ...]) -> bool:
    dispatched_in_order: Final = iter(client_frames)
    json_frames: Final = (
        frame for frame in _client_frames(_frames_of_the_widest_reader(sse_stream)) if _carries_json(frame)
    )
    return all(frame in dispatched_in_order for frame in json_frames)


def anthropic_sse_reads_alike(all_chunks: Sequence[object]) -> bool:
    """Whether every client reads the stream's content as assembly does, so the stream can be forwarded as sent.

    Clients split lines, name frames and place block deltas differently, so the stream needs plain lines, each
    ``event:`` name matching its data ``type`` (an ``error`` frame may carry any data), an empty ``message_start``
    first and nothing after ``message_stop``, and blocks that start at the next index, take their deltas before
    they stop and send their text in order. It must also hold nothing one client reads and another skips: a frame
    the stream never ended, JSON a line-by-line or httpx reader reveals, a delta field assembly reads before the one
    its type names, a ``__json_buf``, or an object Python cannot decode.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks) or ""
    if not _has_only_plain_lines(sse_stream):
        return False
    client_frames: Final = _client_frames(_frames_a_client_dispatches(sse_stream))
    if client_frames != _client_frames(_sse_frames(sse_stream)) or _has_undecodable_object(client_frames):
        return False
    if not (
        _the_widest_reader_dispatches_no_other_json(sse_stream, client_frames)
        and _no_data_line_of_a_frame_is_json_alone(client_frames)
    ):
        return False
    named_data: Final = _named_data(client_frames)
    if not all(_names_its_own_type(name, data) for name, data in named_data):
        return False
    events: Final = tuple(event for event in _client_events(named_data) if event.get("type") != "ping")
    return (
        _opens_with_an_empty_message(events)
        and _ends_at_its_first_message_stop(events)
        and _is_read_by_block(events)
        and all(map(_block_events_read_alike, events))
    )


def _is_error_payload(payload: str) -> bool:
    event: Final = loads_json_object(payload)
    if event is None or not _ERROR_PAYLOAD_KEYS.issuperset(event):
        return False
    return event.get("type") == "error" or ("type" not in event and isinstance(event.get("error"), Mapping))


def _is_error_frame(fields: Sequence[tuple[str, str]]) -> bool:
    match fields:
        case (("data", payload),) | (("event", "error"), ("data", payload)):
            return _is_error_payload(payload)
        case _:
            return False


def is_sse_error_stream(all_chunks: Sequence[object]) -> bool:
    """Whether the buffered stream carries nothing but error frames.

    post_call guardrails run in a chain, so a hook can be handed the terminal error frames an
    earlier guardrail emitted when it blocked. Those carry no message to assemble, and replacing
    them would hide the refusal the client is owed. Covers both wire forms a guardrail emits: the
    Anthropic ``error`` event and the chat-completions ``{"error": ...}`` payload.
    """
    if not all(isinstance(chunk, (str, bytes)) for chunk in all_chunks):
        # A stream mixing typed chunks with an error frame still carries content to scan, and the
        # frames-only join below would drop exactly the part that has to be scanned
        return False
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    frames: Final = tuple(_sse_fields(frame) for frame in _frames_of_the_widest_reader(sse_stream))
    with_data: Final = tuple(fields for fields in frames if not _carries_no_data(fields))
    return len(with_data) > 0 and all(_is_error_frame(fields) for fields in with_data)
