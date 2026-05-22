"""
Pure Python BM25 (Okapi BM25) relevance scorer.

No external dependencies — uses only stdlib.
"""

import math
import re
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens on word boundaries."""
    return re.findall(r"[a-z0-9_]+", text.lower())


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


def bm25_score_messages(
    query: str,
    messages: List[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Score each message's relevance to the query using BM25 (Okapi BM25).

    Parameters:
        query: The reference text to score against (typically the last user message).
        messages: List of message dicts with "content" fields.
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.

    Returns:
        List of float scores, one per message. Higher = more relevant.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return [0.0] * len(messages)

    # Tokenize all documents
    doc_tokens: List[List[str]] = []
    for msg in messages:
        doc_tokens.append(_tokenize(_extract_content(msg)))

    n = len(doc_tokens)
    if n == 0:
        return []

    # Average document length
    doc_lengths = [len(dt) for dt in doc_tokens]
    avgdl = sum(doc_lengths) / n if n > 0 else 1.0

    # Document frequency for each term
    df: Dict[str, int] = {}
    for dt in doc_tokens:
        seen = set(dt)
        for term in seen:
            df[term] = df.get(term, 0) + 1

    # IDF for query terms
    idf: Dict[str, float] = {}
    for term in set(query_terms):
        term_df = df.get(term, 0)
        # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf[term] = math.log((n - term_df + 0.5) / (term_df + 0.5) + 1.0)

    # Build a prefix-expansion map per document: for each query term, find all
    # document tokens that start with that term (min 4 chars match).  This lets
    # "cook" match "cooking" and "auth" match "authentication" without a full
    # stemmer dependency.
    def _expand_tf(query_term: str, tf_counts: Counter) -> int:  # type: ignore[type-arg]
        """Sum TF across all doc tokens that are prefixed by query_term."""
        exact = tf_counts.get(query_term, 0)
        if exact:
            return exact
        if len(query_term) < 4:
            return 0
        return sum(
            count
            for token, count in tf_counts.items()
            if token != query_term and token.startswith(query_term)
        )

    # Score each document
    scores: List[float] = []
    for i, dt in enumerate(doc_tokens):
        if not dt:
            scores.append(0.0)
            continue

        tf_counts = Counter(dt)
        dl = doc_lengths[i]
        score = 0.0

        for term in query_terms:
            if term not in idf:
                continue
            tf = _expand_tf(term, tf_counts)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * numerator / denominator

        scores.append(score)

    return scores
