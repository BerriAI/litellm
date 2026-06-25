"""
Semantic scoring via litellm.embedding().

Computes cosine similarity between the query embedding and each message embedding.
"""

import math
from typing import Any, Dict, List, Optional

from litellm.caching.dual_cache import DualCache


def _extract_content(message: dict) -> str:
    """Extract text content from a message dict."""
    content = message.get("content", "")
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


def _truncate_text(text: str, max_chars: int = 30000) -> str:
    """Truncate long text, keeping first and last portions."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_score_messages(
    query: str,
    messages: List[dict],
    model: str,
    cache: Optional[DualCache] = None,
    embedding_model_params: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Score each message's semantic similarity to the query using embeddings.

    Parameters:
        query: The reference text to score against.
        messages: List of message dicts with "content" fields.
        model: The embedding model to use (e.g., "text-embedding-3-small").
        cache: Optional DualCache for cross-turn embedding caching.
        embedding_model_params: Optional additional kwargs forwarded to
            ``litellm.embedding()``.

    Returns:
        List of float scores (cosine similarity), one per message.
    """
    import litellm

    texts = [_truncate_text(query)]
    for msg in messages:
        texts.append(_truncate_text(_extract_content(msg)))

    # Filter out empty texts — replace with a placeholder to maintain indexing
    processed_texts = [t if t.strip() else "empty" for t in texts]

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": processed_texts,
        "caching": cache is not None,
    }
    if embedding_model_params:
        kwargs = {**kwargs, **embedding_model_params}

    response = litellm.embedding(**kwargs)

    # Extract embedding vectors
    embeddings = [item["embedding"] for item in response.data]

    query_embedding = embeddings[0]
    scores: List[float] = []
    for i in range(1, len(embeddings)):
        scores.append(_cosine_similarity(query_embedding, embeddings[i]))

    return scores
