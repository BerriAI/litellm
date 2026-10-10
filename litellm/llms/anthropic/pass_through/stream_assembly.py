import json
from collections.abc import Sequence
from typing import Final

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
