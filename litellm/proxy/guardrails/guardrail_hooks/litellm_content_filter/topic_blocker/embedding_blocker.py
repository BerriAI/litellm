"""
Embedding-based topic blocker for the LiteLLM content filter.

Uses sentence-transformers to compute semantic similarity between user messages
and denied topics. Two-layer approach:
  - Layer 1: Keyword regex check (~0.1ms) — catches obvious exact matches
  - Layer 2: Embedding cosine similarity (~2-5ms) — catches semantic matches

Usage:
    blocker = EmbeddingTopicBlocker(
        blocked_topics=["investment questions", "stock trading advice"],
        threshold=0.75,
    )
    blocker.check("Should I buy Tesla stock?")  # raises HTTPException 403
    blocker.check("Book me a flight to NYC")    # returns text unchanged
"""

import re
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from litellm._logging import verbose_proxy_logger


class EmbeddingTopicBlocker:
    """
    Embedding-based topic blocker.

    Blocks messages that are semantically similar to any denied topic.
    """

    def __init__(
        self,
        blocked_topics: List[str],
        threshold: float = 0.75,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.blocked_topics = blocked_topics

        # Pre-compute blocked topic embeddings at init time
        self.blocked_embeddings = self.model.encode(
            blocked_topics,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Layer 1: keyword regex patterns for fast exact matching
        self.keyword_patterns = [
            re.compile(rf"\b{re.escape(t.lower())}\b")
            for t in blocked_topics
        ]

    @lru_cache(maxsize=1024)
    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def is_blocked(self, text: str) -> Tuple[bool, Optional[str], float]:
        """
        Check if text matches a blocked topic.

        Returns: (blocked, matched_topic, confidence_score)
        """
        text_lower = text.lower()

        # Layer 1: keyword check (~0.1ms)
        for i, pattern in enumerate(self.keyword_patterns):
            if pattern.search(text_lower):
                return True, self.blocked_topics[i], 1.0

        # Layer 2: embedding similarity (~2-5ms)
        query_embedding = self._embed(text)

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.blocked_embeddings,
        )[0]

        max_idx = np.argmax(similarities)
        max_score = float(similarities[max_idx])

        if max_score >= self.threshold:
            return True, self.blocked_topics[max_idx], max_score

        return False, None, max_score

    def check(self, text: str) -> str:
        """
        Check text against blocked topics.

        Returns the original text if allowed.
        Raises HTTPException(403) if blocked.
        """
        if not text or not text.strip():
            return text

        blocked, matched_topic, score = self.is_blocked(text)

        if blocked:
            verbose_proxy_logger.warning(
                f"Embedding topic blocker: blocked topic '{matched_topic}' "
                f"(score={score:.3f}) for text: {text[:80]}"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Content blocked: denied topic '{matched_topic}'",
                    "topic": matched_topic,
                    "score": score,
                    "match_type": "keyword" if score == 1.0 else "embedding",
                },
            )

        return text
