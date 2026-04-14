"""
Main compress() function — orchestrates BM25/embedding scoring, message stubbing,
and retrieval tool injection.
"""

from typing import Any, Dict, List, Optional, Set, Union, cast

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
from litellm.types.utils import AllMessageValues, Message


def _extract_last_user_message(messages: List[dict]) -> str:
    """Return the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return " ".join(parts)
    return ""


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


def compress(
    messages: List[dict],
    model: str,
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
    if compression_target is None:
        compression_target = compression_trigger * 7 // 10

    original_tokens = token_counter(
        model=model, messages=cast(List[Union[AllMessageValues, Message]], messages)
    )

    # Pass through if below trigger
    if original_tokens <= compression_trigger:
        return CompressedResult(
            messages=messages,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            cache={},
            tools=[],
        )

    # Extract query for relevance scoring
    query = _extract_last_user_message(messages)

    # Score each message
    bm25_scores = bm25_score_messages(query, messages)

    if embedding_model:
        from litellm.compression.scoring.embedding_scorer import (
            embedding_score_messages,
        )

        emb_scores = embedding_score_messages(
            query,
            messages,
            model=embedding_model,
            cache=compression_cache,
            embedding_model_params=embedding_model_params,
        )
        combined_scores = _combine_scores(bm25_scores, emb_scores, bm25_weight=0.4)
    else:
        combined_scores = bm25_scores

    # Sort message indices by score descending
    ranked_indices = sorted(
        range(len(messages)),
        key=lambda i: combined_scores[i],
        reverse=True,
    )

    # Protected messages are never compressed
    protected_indices = _get_protected_indices(messages)
    kept_indices: Set[int] = set(protected_indices)

    # Count tokens for protected messages
    current_tokens = 0
    for i in kept_indices:
        current_tokens += token_counter(
            model=model, text=messages[i].get("content", "") or ""
        )

    # Fill token budget from highest-scoring messages.
    # For each candidate (ranked by relevance):
    #   - If it fits entirely → keep it as-is.
    #   - If it doesn't fit but there's meaningful remaining budget → truncate it
    #     to fill as much of the budget as possible.
    #   - Otherwise → stub it (pointer only, content goes to cache).
    # Multiple messages may be truncated so we preserve partial content from
    # several high-scoring messages rather than fully stubbing all but one.
    truncated_overrides: Dict[int, dict] = {}  # idx -> truncated message dict

    for idx in ranked_indices:
        if idx in kept_indices:
            continue
        msg_content = messages[idx].get("content", "") or ""
        msg_tokens = token_counter(model=model, text=msg_content)
        remaining = compression_target - current_tokens

        if remaining <= 0:
            break  # budget exhausted

        if current_tokens + msg_tokens <= compression_target:
            # Fits entirely
            kept_indices.add(idx)
            current_tokens += msg_tokens
        elif remaining >= 100:
            # Too large to fit whole, but we have budget — truncate it.
            truncated = truncate_message(messages[idx], remaining)
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

    for i, msg in enumerate(messages):
        if i in kept_indices:
            # Use the truncated version if we made one, otherwise the original
            compressed_messages.append(truncated_overrides.get(i, msg))
        else:
            key = extract_key(msg, fallback_index=i, used_keys=used_keys)
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            cache[key] = content
            compressed_messages.append(stub_message(msg, key))

    # Build retrieval tool
    tools = [build_retrieval_tool(list(cache.keys()))] if cache else []

    compressed_tokens = token_counter(
        model=model,
        messages=cast(List[Union[AllMessageValues, Message]], compressed_messages),
    )

    return CompressedResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(1 - (compressed_tokens / original_tokens), 4)
        if original_tokens > 0
        else 0.0,
        cache=cache,
        tools=tools,
    )
