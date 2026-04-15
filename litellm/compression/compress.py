"""
Main compress() function — normalizes input messages, orchestrates BM25/embedding
scoring, message stubbing, and retrieval tool injection.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, cast

from litellm.caching.dual_cache import DualCache
from litellm.compression.message_stubbing import (
    extract_key,
    stub_message,
    truncate_message,
)
from litellm.compression.retrieval_tool import build_retrieval_tool
from litellm.compression.scoring.bm25 import bm25_score_messages
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.compression import CompressedResult, CompressionInputType


def _build_retrieval_tools(keys: List[str], input_type: CompressionInputType) -> List[dict]:
    """
    Build retrieval tool definitions in the target request schema.

    - OpenAI chat completions: keep OpenAI function-tool schema.
    - Anthropic messages: remap OpenAI function-tool schema to Anthropic custom tool.
    """
    if not keys:
        return []

    openai_tools = [build_retrieval_tool(keys)]
    if input_type == "openai_chat_completions":
        return openai_tools

    if input_type == "anthropic_messages":
        # Lazy import to avoid introducing provider transformation imports
        # during module import for non-Anthropic call paths.
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        anthropic_tools, _mcp_servers = AnthropicConfig()._map_tools(openai_tools)
        return cast(List[dict], anthropic_tools)

    return openai_tools


def _content_to_text(content: Any) -> str:
    """
    Convert OpenAI/Anthropic message content blocks to plain text.

    Text extraction policy:
    - Include text-bearing fields only (`text` blocks + string values).
    - For `tool_result`, recurse into nested `content`.
    - Ignore non-textual blocks (images/documents/tool metadata/thinking metadata).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    parts.append(str(part.get("text", "")))
                elif part_type == "tool_result":
                    parts.append(_content_to_text(part.get("content", "")))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""


def _normalize_messages_for_compression(
    messages: List[dict],
    input_type: CompressionInputType,
) -> Tuple[List[dict], List[dict]]:
    """
    Normalize each original message to a text-surrogate content for scoring.

    Returns:
        (normalized_messages, original_messages_copy)
    """
    if input_type not in ("anthropic_messages", "openai_chat_completions"):
        raise ValueError(
            f"Unsupported input_type={input_type}. "
            "Expected 'anthropic_messages' or 'openai_chat_completions'."
        )

    original_messages: List[Dict[str, Any]] = [dict(m) for m in messages]

    normalized_messages: List[dict] = []
    for msg in original_messages:
        normalized_messages.append(
            {
                **msg,
                "content": _content_to_text(msg.get("content", "")),
            }
        )
    return normalized_messages, original_messages


def _extract_last_user_message(messages: List[dict]) -> str:
    """Return the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    return ""


def _extract_tool_use_ids(content: Any) -> List[str]:
    if not isinstance(content, list):
        return []
    tool_use_ids: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "tool_use":
            continue
        tool_use_id = part.get("id")
        if isinstance(tool_use_id, str) and tool_use_id:
            tool_use_ids.append(tool_use_id)
    return tool_use_ids


def _extract_tool_result_ids(content: Any) -> Set[str]:
    if not isinstance(content, list):
        return set()
    tool_result_ids: Set[str] = set()
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
    messages: List[dict],
) -> Tuple[List[Set[int]], Optional[str]]:
    """
    Return atomic 2-message spans for Anthropic tool exchanges.

    Each assistant message containing `tool_use` must be immediately followed by a
    user message containing matching `tool_result` blocks for all tool_use ids.
    """
    spans: List[Set[int]] = []
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


def _get_protected_indices(messages: List[dict]) -> List[int]:
    """
    Return indices of messages that must never be compressed:
    - All system messages
    - The last user message
    - The last assistant message
    """
    protected: List[int] = []

    last_user_idx = None
    last_assistant_idx = None

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "system":
            protected.append(i)
        elif role == "user":
            last_user_idx = i
        elif role == "assistant":
            last_assistant_idx = i

    if last_user_idx is not None:
        protected.append(last_user_idx)
    if last_assistant_idx is not None:
        protected.append(last_assistant_idx)

    return protected


def _combine_scores(
    bm25_scores: List[float],
    emb_scores: List[float],
    bm25_weight: float = 0.4,
) -> List[float]:
    """Weighted average of BM25 and embedding scores, with min-max normalization."""

    def _normalize(scores: List[float]) -> List[float]:
        min_s = min(scores) if scores else 0.0
        max_s = max(scores) if scores else 0.0
        rng = max_s - min_s
        if rng == 0:
            return [0.0] * len(scores)
        return [(s - min_s) / rng for s in scores]

    norm_bm25 = _normalize(bm25_scores)
    norm_emb = _normalize(emb_scores)
    emb_weight = 1.0 - bm25_weight

    return [bm25_weight * b + emb_weight * e for b, e in zip(norm_bm25, norm_emb)]


def _select_kept_indices_for_budget(
    normalized_messages: List[dict],
    original_messages: List[dict],
    combined_scores: List[float],
    compression_target: int,
    model: str,
    initial_kept_indices: Set[int],
    tool_exchange_spans: List[Set[int]],
) -> Tuple[Set[int], Dict[int, dict]]:
    kept_indices = set(initial_kept_indices)
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
    truncated_overrides: Dict[int, dict] = {}  # idx -> truncated message dict
    span_id_by_index: Dict[int, int] = {}
    for span_id, span in enumerate(tool_exchange_spans):
        for idx in span:
            span_id_by_index[idx] = span_id

    # Build single-message candidate units (non-span messages).
    candidate_units: List[Tuple[float, Tuple[int, ...], bool]] = []
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


def _get_dropped_tool_span_indices(
    kept_indices: Set[int], tool_exchange_spans: List[Set[int]]
) -> Set[int]:
    dropped_tool_span_indices: Set[int] = set()
    for span in tool_exchange_spans:
        if not any(idx in kept_indices for idx in span):
            dropped_tool_span_indices.update(span)
    return dropped_tool_span_indices


def compress(
    messages: List[dict],
    model: str,
    input_type: CompressionInputType = "openai_chat_completions",
    compression_trigger: int = 200_000,
    compression_target: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_model_params: Optional[Dict[str, Any]] = None,
    compression_cache: Optional[DualCache] = None,
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
        input_type: Message format of input messages. Must be either:
            - "anthropic_messages"
            - "openai_chat_completions"
            Defaults to "openai_chat_completions" for backward compatibility.
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
    normalized_messages, original_messages = _normalize_messages_for_compression(
        messages=messages,
        input_type=input_type,
    )

    if compression_target is None:
        compression_target = compression_trigger * 7 // 10

    original_tokens = token_counter(
        model=model,
        messages=cast(List[Any], original_messages),
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
    query = _extract_last_user_message(normalized_messages)

    # Score each message
    bm25_scores = bm25_score_messages(query, normalized_messages)

    if embedding_model:
        from litellm.compression.scoring.embedding_scorer import (
            embedding_score_messages,
        )

        emb_scores = embedding_score_messages(
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
    protected_indices = _get_protected_indices(normalized_messages)
    kept_indices: Set[int] = set(protected_indices)

    tool_exchange_spans: List[Set[int]] = []
    if input_type == "anthropic_messages":
        tool_exchange_spans, tool_sequence_error = _extract_anthropic_tool_exchange_spans(
            original_messages
        )
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
    compressed_messages: List[dict] = []
    cache: Dict[str, str] = {}
    used_keys: Set[str] = set()
    dropped_tool_span_indices = _get_dropped_tool_span_indices(
        kept_indices=kept_indices, tool_exchange_spans=tool_exchange_spans
    )

    for i, msg in enumerate(original_messages):
        if i in dropped_tool_span_indices:
            continue
        if i in kept_indices:
            # Use the truncated version if we made one, otherwise the original
            compressed_messages.append(truncated_overrides.get(i, msg))
        else:
            key = extract_key(
                normalized_messages[i], fallback_index=i, used_keys=used_keys
            )
            content = _content_to_text(msg.get("content", ""))
            cache[key] = content
            compressed_messages.append(stub_message(msg, key))

    # Build retrieval tool in the target request schema
    tools = _build_retrieval_tools(list(cache.keys()), input_type=input_type)

    compressed_tokens = token_counter(
        model=model,
        messages=cast(List[Any], compressed_messages),
    )

    return CompressedResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=(
            round(1 - (compressed_tokens / original_tokens), 4)
            if original_tokens > 0
            else 0.0
        ),
        cache=cache,
        tools=tools,
    )
