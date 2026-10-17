import codecs
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from itertools import groupby
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
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
    """Build a logged response while preserving native usage and finish reasons.

    Pure-text streams keep their existing fast path. Complete mixed streams collapse
    adjacent text, signed thinking and tool argument deltas before native conversion.
    Incomplete or unfamiliar mixed streams retain the original reconstruction.
    """
    collapsed: Final = collapse_pure_text_chunks(all_chunks)
    if collapsed is not None:
        return build_complete_streaming_response_legacy(
            all_chunks=collapsed,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            speed=speed,
        )
    compacted: Final = compact_content_block_deltas(all_chunks)
    return build_complete_streaming_response_legacy(
        all_chunks=all_chunks if compacted is None else compacted,
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


_DELTA_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}
)
_BLOCK_DELTAS: Final[Mapping[str, str]] = MappingProxyType(
    {"text": "text_delta", "thinking": "thinking_delta", "tool_use": "input_json_delta"}
)
_SSE_OBJECT: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class _CompactionEvent:
    raw: str
    data: Mapping[str, object]
    delta: tuple[int, str, str] | None


@dataclass(frozen=True, slots=True)
class _CompactionState:
    active: tuple[int, str] | None = None
    signed_thinking: bool = False
    started: bool = False
    stopped: bool = False


def _compaction_event(raw: str) -> _CompactionEvent | None:
    lines: Final = tuple(line.strip() for line in raw.splitlines() if line.strip())
    payloads: Final = tuple(line[5:].strip() for line in lines if line.startswith("data:"))
    if len(payloads) != 1 or any(not line.startswith(("event:", "data:")) for line in lines):
        return None
    data: Final = _SSE_OBJECT.validate_json(payloads[0])
    kind: Final = data.get("type")
    if any(line.startswith("event:") and line[6:].strip() != kind for line in lines):
        return None
    delta: Final = data.get("delta")
    if kind != "content_block_delta" or not isinstance(delta, Mapping):
        return _CompactionEvent(raw, data, None)
    fields: Final = _SSE_OBJECT.validate_python(delta)
    delta_type: Final = fields.get("type")
    index: Final = data.get("index")
    if not isinstance(delta_type, str) or delta_type not in _DELTA_FIELDS:
        return _CompactionEvent(raw, data, None)
    field: Final = _DELTA_FIELDS[delta_type]
    part: Final = fields.get(field)
    if (
        type(index) is not int
        or index < 0
        or set(data) != {"type", "index", "delta"}
        or set(fields) != {"type", field}
        or not isinstance(part, str)
    ):
        return None
    return _CompactionEvent(raw, data, (index, delta_type, part))


def _start_compaction_block(state: _CompactionState, event: _CompactionEvent) -> _CompactionState | None:
    block: Final = event.data.get("content_block")
    index: Final = event.data.get("index")
    if state.active is not None or not isinstance(block, Mapping) or type(index) is not int or index < 0:
        return None
    block_fields: Final = _SSE_OBJECT.validate_python(block)
    block_type: Final = block_fields.get("type")
    if not isinstance(block_type, str) or block_type not in _BLOCK_DELTAS:
        return None
    return replace(state, active=(index, _BLOCK_DELTAS[block_type]), signed_thinking=False)


def _advance_compaction_delta(state: _CompactionState, event: _CompactionEvent) -> _CompactionState | None:
    if state.active is None or event.data.get("index") != state.active[0]:
        return None
    if event.delta is not None:
        return state if not state.signed_thinking and event.delta[1] == state.active[1] else None
    delta: Final = event.data.get("delta")
    if not isinstance(delta, Mapping) or state.active[1] != "thinking_delta":
        return None
    delta_fields: Final = _SSE_OBJECT.validate_python(delta)
    signature: Final = delta_fields.get("signature")
    if delta_fields.get("type") != "signature_delta" or not isinstance(signature, str) or not signature:
        return None
    return replace(state, signed_thinking=True)


def _stop_compaction_block(state: _CompactionState, event: _CompactionEvent) -> _CompactionState | None:
    if state.active is None or event.data.get("index") != state.active[0]:
        return None
    if state.active[1] == "thinking_delta" and not state.signed_thinking:
        return None
    return replace(state, active=None)


def _state_after_event(state: _CompactionState | None, event: _CompactionEvent) -> _CompactionState | None:
    if state is None or state.stopped:
        return None
    kind: Final = event.data.get("type")
    if not state.started:
        return replace(state, started=True) if kind == "message_start" else None
    if kind == "message_start":
        return None
    if kind == "content_block_start":
        return _start_compaction_block(state, event)
    if kind == "content_block_delta":
        return _advance_compaction_delta(state, event)
    if kind == "content_block_stop":
        return _stop_compaction_block(state, event)
    if kind == "message_stop":
        return replace(state, stopped=True) if state.active is None else None
    if kind in ("message_delta", "ping"):
        return state if kind == "ping" or state.active is None else None
    return None


def _raw_sse_events(all_chunks: Sequence[str | bytes]) -> Iterator[str]:
    for raw in all_chunks:
        yield from split_sse_chunk_into_events(raw)


def _delta_key(event: _CompactionEvent) -> tuple[int, str] | None:
    return None if event.delta is None else event.delta[:2]


def _compact_event_run(run: tuple[_CompactionEvent, ...]) -> Iterator[str]:
    first: Final = run[0]
    if len(run) == 1 or first.delta is None:
        yield from (event.raw for event in run)
        return
    index, delta_type, _ = first.delta
    field: Final = _DELTA_FIELDS[delta_type]
    content: Final = "".join(event.delta[2] for event in run if event.delta is not None)
    payload: Final = {"type": "content_block_delta", "index": index, "delta": {"type": delta_type, field: content}}
    yield "event: content_block_delta\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _compacted_events(events: tuple[_CompactionEvent, ...]) -> Iterator[str]:
    for _, run in groupby(events, key=_delta_key):
        yield from _compact_event_run(tuple(run))


def compact_content_block_deltas(all_chunks: Sequence[str | bytes]) -> tuple[str, ...] | None:
    """Collapse adjacent deltas without changing signed blocks or usage events."""
    try:
        parsed: Final = tuple(_compaction_event(event) for event in _raw_sse_events(all_chunks))
    except (ValueError, TypeError, UnicodeError, AttributeError):
        return None
    if any(event is None for event in parsed):
        return None
    events: Final = tuple(event for event in parsed if event is not None)
    initial: Final[_CompactionState | None] = _CompactionState()
    state: Final = reduce(_state_after_event, events, initial)
    if state is None or not state.stopped or state.active is not None:
        return None
    compacted: Final = tuple(_compacted_events(events))
    return compacted if len(compacted) < len(events) else None


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


def extract_sse_data(event_str: str) -> dict | None:
    """Parse the JSON object from the ``data:`` line of an Anthropic SSE event."""
    for line in event_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            payload = stripped[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                return None
            try:
                return json.loads(payload)
            except (ValueError, TypeError):
                return None
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


def is_raw_sse_stream(all_chunks: Sequence[object]) -> bool:
    return any(isinstance(chunk, (str, bytes)) for chunk in all_chunks)


def _joined_sse_stream(all_chunks: Sequence[object]) -> str | None:
    raw: Final = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        for chunk in all_chunks
        if isinstance(chunk, (str, bytes))
    )
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
    except UnicodeDecodeError:
        return None


def _parsed_sse_events(sse_stream: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        event_data
        for event in split_sse_chunk_into_events(sse_stream)
        if (event_data := extract_sse_data(event)) is not None
    )


def _anthropic_message_start(sse_stream: str) -> Mapping[str, object] | None:
    return next(
        (
            message
            for event_data in _parsed_sse_events(sse_stream)
            if event_data.get("type") == "message_start" and isinstance(message := event_data.get("message"), dict)
        ),
        None,
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
    return any(event.get("type") in _ANTHROPIC_EVENT_TYPES for event in _parsed_sse_events(sse_stream))


def assemble_anthropic_sse_stream(
    all_chunks: Sequence[object], *, restore_identity: bool = False
) -> ModelResponse | None:
    """Assemble raw Anthropic SSE frames into a ModelResponse.

    ``restore_identity`` stamps the upstream message id and model onto the result, which the
    assembler does not carry through. It is off by default so callers that re-emit the assembled
    response keep the wire shape they had before this helper was shared. The writes land on a
    freshly built object that is unreachable from caller state until returned.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    message_start: Final = _anthropic_message_start(sse_stream)
    if message_start is None:
        return None
    model: Final = message_start.get("model") if restore_identity else None
    try:
        assembled: Final = build_complete_streaming_response(
            all_chunks=(sse_stream,),
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
    events: Final = _parsed_sse_events(sse_stream)
    return len(events) > 0 and all(
        event.get("type") == "error" or isinstance(event.get("error"), Mapping) for event in events
    )
