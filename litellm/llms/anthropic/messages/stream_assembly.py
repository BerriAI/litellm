import codecs
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, map_finish_reason
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

    text_block_indexes: Final[set] = set()
    out: Final[list[str]] = []
    pending_text: list[str] = []
    pending_index: int | None = None
    saw_any_text_delta = False

    def flush() -> None:
        nonlocal pending_text, pending_index
        if pending_text:
            merged: Final = {
                "type": "content_block_delta",
                "index": pending_index if pending_index is not None else 0,
                "delta": {"type": "text_delta", "text": "".join(pending_text)},
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
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

        etype = data.get("type")
        if etype == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") != "text":
                return None
            text_block_indexes.add(data.get("index"))
            flush()
            out.append(ev)
        elif etype == "content_block_delta":
            delta = data.get("delta") or {}
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


def _has_non_object_data(event_str: str) -> bool:
    data_start: Final = event_str.find("data:")
    return data_start != -1 and not event_str[data_start + len("data:") :].lstrip().startswith("{")


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
                if _has_non_object_data(event_str):
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
    if payload is None or not payload.startswith("{"):
        return None
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


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


def _sse_event_data(event: str) -> Mapping[str, object] | None:
    return as_str_mapping(extract_sse_data(event))


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


def _json_mapping(payload: str) -> Mapping[str, object] | None:
    try:
        value: Final[object] = json.loads(payload)
    except ValueError:
        return None
    return as_str_mapping(value)


def _parsed_sse_events(sse_stream: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        event_data
        for event in split_sse_chunk_into_events(sse_stream)
        if (event_data := _sse_event_data(event)) is not None
    )


def _anthropic_message_start(sse_stream: str) -> Mapping[str, object] | None:
    return next(
        (
            message
            for event_data in _parsed_sse_events(sse_stream)
            if event_data.get("type") == "message_start"
            and (message := as_str_mapping(event_data.get("message"))) is not None
        ),
        None,
    )


def anthropic_message_delta_fields(all_chunks: Sequence[object]) -> Mapping[str, object]:
    """The ``stop_details`` and ``safeguard_results`` of the stream's ``message_delta``.

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
        value: Final[object] = json.loads(payload)
    except ValueError:
        return False
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
    one_line: Final = payload.translate(_JSON_ESCAPED_LINE_BREAKS) if _json_mapping(payload) is not None else payload
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
    """The blocks the stream opened and never stopped, which only a failed upstream leaves."""
    return _unfinished_blocks(_parsed_sse_events(_joined_sse_stream(all_chunks) or ""))


def _sse_events_with_data(sse_stream: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return tuple(
        (event, event_data)
        for event in split_sse_chunk_into_events(sse_stream)
        if (event_data := _sse_event_data(event)) is not None
    )


def _content_events_before_upstream_error(sse_stream: str) -> tuple[str, ...]:
    """The stream's events minus its upstream ``error`` and every cut-off block assembly cannot read.

    A cut-off text block is still text the client was sent, so it stays. A cut-off tool call's
    input is a JSON prefix, and a cut-off thinking block has no signature, so neither is a block the
    client can act on, and both are left out.
    """
    if not _error_events(sse_stream):
        return (sse_stream,)
    events: Final = _sse_events_with_data(sse_stream)
    cut_off: Final = frozenset(
        index
        for index, block in _unfinished_blocks(tuple(event_data for _, event_data in events)).items()
        if block.get("type") != "text"
    )
    return tuple(
        event
        for event, event_data in events
        if event_data.get("type") != "error" and _block_index(event_data) not in cut_off
    )


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
    message_start: Final = _anthropic_message_start(sse_stream)
    if message_start is None:
        return None
    model: Final = message_start.get("model") if restore_identity else None
    content_events: Final = (
        _content_events_before_upstream_error(sse_stream) if through_upstream_error else (sse_stream,)
    )
    try:
        assembled: Final = build_complete_streaming_response(
            all_chunks=content_events,
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


def _is_error_payload(payload: str) -> bool:
    event: Final = _json_mapping(payload)
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
