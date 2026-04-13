"""
Main compress() function — orchestrates BM25/embedding scoring, message stubbing,
and retrieval tool injection.
"""

from typing import Any, Dict, List, Literal, Optional, Set, Union, cast

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
from litellm.types.llms.anthropic import AllAnthropicMessageValues
from litellm.types.llms.openai import AllMessageValues

CompressionInputType = Literal["openai_chat_completions", "anthropic_messages"]


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            tool_content = part.get("content")
            if isinstance(tool_content, str):
                parts.append(tool_content)
            elif isinstance(tool_content, list):
                for tool_part in tool_content:
                    if isinstance(tool_part, str):
                        parts.append(tool_part)
                    elif isinstance(tool_part, dict):
                        nested_text = tool_part.get("text")
                        if isinstance(nested_text, str):
                            parts.append(nested_text)
        return " ".join(parts)
    return ""


def _normalize_messages_for_compression(
    messages: Union[List[AllMessageValues], List[AllAnthropicMessageValues]],
) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        msg_dict = cast(Dict[str, Any], message)
        normalized_messages.append(
            {
                "role": msg_dict.get("role", ""),
                "content": _extract_text_content(msg_dict.get("content", "")),
            }
        )
    return normalized_messages


def _remap_compressed_messages(
    original_messages: Union[List[AllMessageValues], List[AllAnthropicMessageValues]],
    normalized_original_messages: List[Dict[str, Any]],
    normalized_compressed_messages: List[Dict[str, Any]],
    input_type: CompressionInputType,
) -> List[dict]:
    if input_type == "openai_chat_completions":
        return normalized_compressed_messages

    remapped_messages: List[dict] = []
    for idx, original_message in enumerate(original_messages):
        original_msg = cast(Dict[str, Any], original_message)
        remapped_message = {**original_msg}
        original_content = original_msg.get("content", "")
        original_normalized_content = normalized_original_messages[idx].get(
            "content", ""
        )
        compressed_content = normalized_compressed_messages[idx].get("content", "")

        if isinstance(original_content, list):
            if compressed_content == original_normalized_content:
                remapped_message["content"] = original_content
            else:
                remapped_message["content"] = [
                    {"type": "text", "text": compressed_content}
                ]
        else:
            remapped_message["content"] = compressed_content

        remapped_messages.append(remapped_message)

    return remapped_messages


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Return the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text_content(msg.get("content", ""))
    return ""


def _get_protected_indices(messages: List[Dict[str, Any]]) -> List[int]:
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


def compress(
    messages: Union[List[AllMessageValues], List[AllAnthropicMessageValues]],
    model: str,
    input_type: CompressionInputType,
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
    and the lowest-relevance messages are replaced with stubs. Originals are
    cached and a retrieval tool is injected so the model can recover dropped
    content on demand.

    Parameters:
        messages: The conversation messages to (potentially) compress.
        model: The LLM model name — used for token counting.
        input_type: Source format for messages.
            One of: ``openai_chat_completions`` or ``anthropic_messages``.
        compression_trigger: Only compress if input exceeds this token count.
        compression_target: Target token count after compression.
            Defaults to ``compression_trigger * 7 // 10``.
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
    if input_type not in ("openai_chat_completions", "anthropic_messages"):
        raise ValueError(
            "Invalid input_type. Expected 'openai_chat_completions' or 'anthropic_messages'."
        )

    if compression_target is None:
        compression_target = compression_trigger * 7 // 10

    normalized_messages = _normalize_messages_for_compression(messages=messages)
    original_tokens = token_counter(
        model=model,
        messages=cast(List[Any], normalized_messages),
    )

    # Pass through if below trigger
    if original_tokens <= compression_trigger:
        return CompressedResult(
            messages=cast(List[dict], messages),
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            cache={},
            tools=[],
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

    # Sort message indices by score descending
    ranked_indices = sorted(
        range(len(normalized_messages)),
        key=lambda i: combined_scores[i],
        reverse=True,
    )

    # Protected messages are never compressed
    protected_indices = _get_protected_indices(normalized_messages)
    kept_indices: Set[int] = set(protected_indices)

    # Count tokens for protected messages
    current_tokens = 0
    for i in kept_indices:
        current_tokens += token_counter(
            model=model, text=normalized_messages[i].get("content", "") or ""
        )

    # Fill token budget from highest-scoring messages.
    truncated_overrides: Dict[int, dict] = {}  # idx -> truncated message dict

    for idx in ranked_indices:
        if idx in kept_indices:
            continue
        msg_content = normalized_messages[idx].get("content", "") or ""
        msg_tokens = token_counter(model=model, text=msg_content)
        remaining = compression_target - current_tokens

        if remaining <= 0:
            break  # budget exhausted

        if current_tokens + msg_tokens <= compression_target:
            kept_indices.add(idx)
            current_tokens += msg_tokens
        elif remaining >= 100:
            truncated = truncate_message(normalized_messages[idx], remaining)
            truncated_tokens = token_counter(
                model=model,
                text=truncated.get("content", "") or "",
            )
            truncated_overrides[idx] = truncated
            kept_indices.add(idx)
            current_tokens += truncated_tokens

    # Build compressed messages and cache
    compressed_messages: List[dict] = []
    cache: Dict[str, str] = {}
    used_keys: Set[str] = set()

    for i, msg in enumerate(normalized_messages):
        if i in kept_indices:
            compressed_messages.append(truncated_overrides.get(i, msg))
        else:
            key = extract_key(msg, fallback_index=i, used_keys=used_keys)
            cache[key] = msg.get("content", "")
            compressed_messages.append(stub_message(msg, key))

    remapped_compressed_messages = _remap_compressed_messages(
        original_messages=messages,
        normalized_original_messages=normalized_messages,
        normalized_compressed_messages=compressed_messages,
        input_type=input_type,
    )

    tools = [build_retrieval_tool(list(cache.keys()))] if cache else []
    compressed_tokens = token_counter(
        model=model,
        messages=cast(List[Any], compressed_messages),
    )

    return CompressedResult(
        messages=remapped_compressed_messages,
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
