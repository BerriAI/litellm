"""
Main compress() function — normalizes input messages, orchestrates BM25/embedding
scoring, message stubbing, and retrieval tool injection.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from litellm.caching.dual_cache import DualCache
from litellm.compression.message_stubbing import (
    extract_key,
    stub_message,
    truncate_message,
)
from litellm.compression.retrieval_tool import build_retrieval_tool
from litellm.compression.scoring.bm25 import bm25_score_messages
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.compression import CompressedResult
from litellm.types.utils import CallTypes

# CallTypes that produce Anthropic-shaped messages (structured content blocks).
# Everything else is treated as OpenAI chat-completions shape.
_ANTHROPIC_CALL_TYPES: Final = frozenset({CallTypes.anthropic_messages.value})
# CallTypes that are valid targets for compression.  Compression operates on
# message-shaped inputs, so we only accept call types whose payload is a list
# of role/content messages.
_SUPPORTED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.completion.value,
        CallTypes.acompletion.value,
        CallTypes.anthropic_messages.value,
    }
)


def _normalize_call_type(call_type: CallTypes | str) -> str:
    """Return the string value for a ``CallTypes`` enum or a raw string."""
    if isinstance(call_type, CallTypes):
        return call_type.value
    return call_type


def _is_anthropic_call_type(call_type: str) -> bool:
    return call_type in _ANTHROPIC_CALL_TYPES


def _build_retrieval_tools(keys: list[str], call_type: str) -> list[dict]:
    """
    Build retrieval tool definitions in the target request schema.

    - Chat-completions call types: keep OpenAI function-tool schema.
    - Anthropic messages call type: remap to Anthropic's custom tool schema.
    """
    if not keys:
        return []

    openai_tools: Final = [build_retrieval_tool(keys)]
    if not _is_anthropic_call_type(call_type):
        return openai_tools

    # Lazy import to avoid introducing provider transformation imports during
    # module import for non-Anthropic call paths.
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    anthropic_tools, _mcp_servers = AnthropicConfig()._map_tools(openai_tools)
    return cast(list[dict], anthropic_tools)


def _content_to_text(content: Any) -> str:
    """
    Convert OpenAI/Anthropic message content blocks to plain text.

    Text extraction policy:
    - Include text-bearing fields only (`text` blocks + string values).
    - For `tool_result`, expand into nested `content` items.
    - Ignore non-textual blocks (images/documents/tool metadata/thinking metadata).

    Implemented iteratively (stack-based) to avoid unbounded recursion.
    """
    parts: Final[list[str]] = []
    stack: Final[list[Any]] = [content]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            # Push list items in reverse order so they are processed left-to-right.
            for element in reversed(item):
                stack.append(element)
        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "tool_result":
                stack.append(item.get("content", ""))
    return " ".join(parts)


def _normalize_messages_for_compression(
    messages: list[dict],
    call_type: str,
) -> tuple[list[dict], list[dict]]:
    """
    Normalize each original message to a text-surrogate content for scoring.

    Returns:
        (normalized_messages, original_messages_copy)
    """
    if call_type not in _SUPPORTED_CALL_TYPES:
        raise ValueError(
            f"Unsupported call_type={call_type!r} for compression. Expected one of: {sorted(_SUPPORTED_CALL_TYPES)}."
        )

    original_messages: Final[list[dict[str, Any]]] = [dict(m) for m in messages]

    normalized_messages: Final[list[dict]] = []
    for msg in original_messages:
        normalized_messages.append(
            {
                **msg,
                "content": _content_to_text(msg.get("content", "")),
            }
        )
    return normalized_messages, original_messages


def _extract_last_user_message(messages: list[dict]) -> str:
    """Return the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    return ""


def _extract_tool_use_ids(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    tool_use_ids: Final[list[str]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "tool_use":
            continue
        tool_use_id = part.get("id")
        if isinstance(tool_use_id, str) and tool_use_id:
            tool_use_ids.append(tool_use_id)
    return tool_use_ids


def _extract_tool_result_ids(content: Any) -> set[str]:
    if not isinstance(content, list):
        return set()
    tool_result_ids: Final[set[str]] = set()
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "tool_result":
            continue
        tool_use_id = part.get("tool_use_id")
        if isinstance(tool_use_id, str) and tool_use_id:
            tool_result_ids.add(tool_use_id)
    return tool_result_ids


def _extract_anthropic_tool_exchange_spans(
    messages: list[dict],
) -> tuple[list[set[int]], str | None]:
    """
    Return atomic 2-message spans for Anthropic tool exchanges.

    Each assistant message containing `tool_use` must be immediately followed by a
    user message containing matching `tool_result` blocks for all tool_use ids.
    """
    spans: Final[list[set[int]]] = []
    i = 0
    while i < len(messages):
        current = messages[i]
        if current.get("role") != "assistant":
            i += 1
            continue

        tool_use_ids = _extract_tool_use_ids(current.get("content"))
        if not tool_use_ids:
            i += 1
            continue

        if i + 1 >= len(messages):
            return [], "invalid_anthropic_tool_sequence"

        next_msg = messages[i + 1]
        if next_msg.get("role") != "user":
            return [], "invalid_anthropic_tool_sequence"

        tool_result_ids = _extract_tool_result_ids(next_msg.get("content"))
        if not tool_result_ids:
            return [], "invalid_anthropic_tool_sequence"

        for tool_use_id in tool_use_ids:
            if tool_use_id not in tool_result_ids:
                return [], "invalid_anthropic_tool_sequence"

        spans.append({i, i + 1})
        i += 2

    return spans, None


def get_protected_indices(messages: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
    """
    Return indices of messages that must never be compressed:
    - All system messages
    - The last user message
    - The last assistant message

    The last user message is what the model is being asked to act on right now,
    so compressing it replaces the live instruction with a marker. Compression
    guardrails share this policy; see the Headroom guardrail.
    """
    system_indices: Final = tuple(index for index, msg in enumerate(messages) if msg.get("role", "") == "system")
    last_user: Final = tuple(index for index, msg in enumerate(messages) if msg.get("role", "") == "user")[-1:]
    last_assistant = tuple(index for index, msg in enumerate(messages) if msg.get("role", "") == "assistant")[-1:]
    return system_indices + last_user + last_assistant


def _combine_scores(
    bm25_scores: list[float],
    emb_scores: list[float],
    bm25_weight: float = 0.4,
) -> list[float]:
    """Weighted average of BM25 and embedding scores, with min-max normalization."""

    def _normalize(scores: list[float]) -> list[float]:
        min_s: Final = min(scores) if scores else 0.0
        max_s: Final = max(scores) if scores else 0.0
        rng: Final = max_s - min_s
        if rng == 0:
            return [0.0] * len(scores)
        return [(s - min_s) / rng for s in scores]

    norm_bm25: Final = _normalize(bm25_scores)
    norm_emb: Final = _normalize(emb_scores)
    emb_weight: Final = 1.0 - bm25_weight

    return [bm25_weight * b + emb_weight * e for b, e in zip(norm_bm25, norm_emb)]


def _select_kept_indices_for_budget(
    normalized_messages: list[dict],
    original_messages: list[dict],
    combined_scores: list[float],
    compression_target: int,
    model: str,
    initial_kept_indices: set[int],
    tool_exchange_spans: list[set[int]],
) -> tuple[set[int], dict[int, dict]]:
    kept_indices: Final = set(initial_kept_indices)
    current_tokens = 0
    for i in kept_indices:
        current_tokens += token_counter(
            model=model,
            text=cast(str, normalized_messages[i].get("content", "") or ""),
        )

    # Fill token budget from highest-scoring units.
    # A unit is either:
    # 1) a single message index, or
    # 2) an Anthropic tool-exchange span that must be kept/dropped atomically.
    truncated_overrides: Final[dict[int, dict]] = {}  # idx -> truncated message dict
    span_id_by_index: Final[dict[int, int]] = {}
    for span_id, span in enumerate(tool_exchange_spans):
        for idx in span:
            span_id_by_index[idx] = span_id

    # Build single-message candidate units (non-span messages).
    candidate_units: Final[list[tuple[float, tuple[int, ...], bool]]] = []
    for idx in range(len(normalized_messages)):
        if idx in span_id_by_index or idx in kept_indices:
            continue
        candidate_units.append((combined_scores[idx], (idx,), True))

    # Build span candidate units (atomic keep/drop for tool exchanges).
    for span in tool_exchange_spans:
        span_indices = tuple(sorted(span))
        if any(idx in kept_indices for idx in span_indices):
            continue
        span_score = max(combined_scores[idx] for idx in span_indices)
        candidate_units.append((span_score, span_indices, False))

    # Sort by descending relevance score.
    candidate_units.sort(key=lambda item: item[0], reverse=True)

    for _score, indices, can_truncate in candidate_units:
        if any(idx in kept_indices for idx in indices):
            continue
        msg_tokens = 0
        for idx in indices:
            msg_tokens += token_counter(
                model=model,
                text=cast(str, normalized_messages[idx].get("content", "") or ""),
            )
        remaining = compression_target - current_tokens

        if remaining <= 0:
            break  # budget exhausted

        if current_tokens + msg_tokens <= compression_target:
            # Fits entirely
            kept_indices.update(indices)
            current_tokens += msg_tokens
        elif can_truncate and len(indices) == 1 and remaining >= 100:
            # Too large to fit whole single message, but we have budget — truncate it.
            idx = indices[0]
            truncated = truncate_message(original_messages[idx], remaining)
            truncated_tokens = token_counter(
                model=model,
                text=truncated.get("content", "") or "",
            )
            truncated_overrides[idx] = truncated
            kept_indices.add(idx)
            current_tokens += truncated_tokens

    return kept_indices, truncated_overrides


def _get_dropped_tool_span_indices(kept_indices: set[int], tool_exchange_spans: list[set[int]]) -> set[int]:
    dropped_tool_span_indices: Final[set[int]] = set()
    for span in tool_exchange_spans:
        if not any(idx in kept_indices for idx in span):
            dropped_tool_span_indices.update(span)
    return dropped_tool_span_indices


def compress(
    messages: list[dict],
    model: str,
    call_type: CallTypes | str = CallTypes.completion,
    compression_trigger: int = 200_000,
    compression_target: int | None = None,
    embedding_model: str | None = None,
    embedding_model_params: dict[str, Any] | None = None,
    compression_cache: DualCache | None = None,
) -> CompressedResult:
    """
    Compress a list of messages by replacing low-relevance content with stubs.

    Messages below ``compression_trigger`` tokens pass through unchanged.
    Messages above are scored with BM25 (and optionally embeddings), ranked,
    and the lowest-relevance messages are replaced with stubs.  Originals are
    cached and a retrieval tool is injected so the model can recover dropped
    content on demand.

    Parameters:
        messages: The conversation messages to (potentially) compress.
        model: The LLM model name — used for token counting.
        call_type: The LiteLLM call type whose message schema these messages
            follow.  Supported values:
            - ``CallTypes.completion`` / ``CallTypes.acompletion`` — OpenAI
              chat-completions shape (default)
            - ``CallTypes.anthropic_messages`` — Anthropic Messages shape
              (structured content blocks + atomic tool exchanges)
        compression_trigger: Only compress if input exceeds this token count.
        compression_target: Target token count after compression.
            Defaults to ``compression_trigger // 2``.
        embedding_model: If provided, use BM25 + embeddings for scoring.
            If ``None``, BM25 only.
        embedding_model_params: Optional kwargs forwarded to
            ``litellm.embedding()`` when ``embedding_model`` is set.
        compression_cache: Passed through to ``litellm.embedding()`` for
            cross-turn caching of embedding vectors.

    Returns:
        A ``CompressedResult`` dict containing compressed messages, token
        counts, a cache of original content, and the retrieval tool definition.
    """
    call_type_str: Final = _normalize_call_type(call_type)
    normalized_messages, original_messages = _normalize_messages_for_compression(
        messages=messages,
        call_type=call_type_str,
    )

    if compression_target is None:
        compression_target = compression_trigger * 7 // 10

    original_tokens: Final = token_counter(
        model=model,
        messages=cast(list[Any], original_messages),
    )

    # Pass through if below trigger
    if original_tokens <= compression_trigger:
        return CompressedResult(
            messages=original_messages,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            cache={},
            tools=[],
            compression_skipped_reason="below_trigger",
        )

    # Extract query for relevance scoring
    query: Final = _extract_last_user_message(normalized_messages)

    # Score each message
    bm25_scores: Final = bm25_score_messages(query, normalized_messages)

    if embedding_model:
        from litellm.compression.scoring.embedding_scorer import (
            embedding_score_messages,
        )

        emb_scores: Final = embedding_score_messages(
            query,
            normalized_messages,
            model=embedding_model,
            cache=compression_cache,
            embedding_model_params=embedding_model_params,
        )
        combined_scores = _combine_scores(bm25_scores, emb_scores, bm25_weight=0.4)
    else:
        combined_scores = bm25_scores

    # Protected messages are never compressed
    protected_indices: Final = get_protected_indices(normalized_messages)
    kept_indices: set[int] = set(protected_indices)

    tool_exchange_spans: list[set[int]] = []
    if _is_anthropic_call_type(call_type_str):
        tool_exchange_spans, tool_sequence_error = _extract_anthropic_tool_exchange_spans(original_messages)
        if tool_sequence_error is not None:
            return CompressedResult(
                messages=original_messages,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                cache={},
                tools=[],
                compression_skipped_reason=tool_sequence_error,
            )

        for span in tool_exchange_spans:
            # If any message in the span is protected, keep the whole span.
            if any(idx in kept_indices for idx in span):
                kept_indices.update(span)

    kept_indices, truncated_overrides = _select_kept_indices_for_budget(
        normalized_messages=normalized_messages,
        original_messages=original_messages,
        combined_scores=combined_scores,
        compression_target=compression_target,
        model=model,
        initial_kept_indices=kept_indices,
        tool_exchange_spans=tool_exchange_spans,
    )

    # Build compressed messages and cache
    compressed_messages: Final[list[dict]] = []
    cache: Final[dict[str, str]] = {}
    used_keys: Final[set[str]] = set()
    dropped_tool_span_indices: Final = _get_dropped_tool_span_indices(
        kept_indices=kept_indices, tool_exchange_spans=tool_exchange_spans
    )

    for i, msg in enumerate(original_messages):
        if i in dropped_tool_span_indices:
            continue
        if i in kept_indices:
            # Use the truncated version if we made one, otherwise the original
            compressed_messages.append(truncated_overrides.get(i, msg))
        else:
            key = extract_key(normalized_messages[i], fallback_index=i, used_keys=used_keys)
            content = _content_to_text(msg.get("content", ""))
            cache[key] = content
            compressed_messages.append(stub_message(msg, key))

    # Build retrieval tool in the target request schema
    tools: Final = _build_retrieval_tools(list(cache.keys()), call_type=call_type_str)

    compressed_tokens: Final = token_counter(
        model=model,
        messages=cast(list[Any], compressed_messages),
    )

    return CompressedResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=(round(1 - (compressed_tokens / original_tokens), 4) if original_tokens > 0 else 0.0),
        cache=cache,
        tools=tools,
    )
