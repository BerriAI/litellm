"""
Simple topic blocker for the LiteLLM content filter.

Blocks messages that match denied topics. A topic is defined by:
  - identifier_words: words that signal the topic (e.g. "stock", "invest")
  - block_words: words that combined with an identifier = block (e.g. "buy", "price")
  - always_block_phrases: phrases that are always blocked regardless of context
  - exception_phrases: phrases that override a match (e.g. "in stock")

Matching rules:
  - A message is blocked if any SENTENCE contains both an identifier word
    AND a block word (conditional match).
  - A message is blocked if it contains any always-block phrase.
  - Exception phrases override conditional matches in the sentence they appear in.
  - All matching uses word boundaries (whole words only).
  - Text is normalized before matching (lowercase, strip zero-width chars, normalize unicode).
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger

# Characters to strip before matching
_ZERO_WIDTH_CHARS = re.compile(
    "["
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u200e"  # left-to-right mark
    "\u200f"  # right-to-left mark
    "\u2060"  # word joiner
    "\ufeff"  # zero-width no-break space
    "]"
)

# Sentence splitting: split on . ! ? ; and newlines
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")


def _normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip zero-width chars, normalize unicode."""
    # Strip zero-width characters
    text = _ZERO_WIDTH_CHARS.sub("", text)
    # Normalize unicode (fullwidth -> ASCII, accented -> base, homoglyphs -> latin)
    text = unicodedata.normalize("NFKD", text)
    # Drop combining marks (accents etc.) so "café" → "cafe", "αlpha" stays "αlpha"
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    return text


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """Build a regex that matches a phrase at word boundaries."""
    escaped = re.escape(phrase.lower())
    return re.compile(r"\b" + escaped + r"\b")


class DeniedTopic:
    """A single denied topic with its matching configuration."""

    def __init__(
        self,
        topic_name: str,
        identifier_words: List[str],
        block_words: List[str],
        always_block_phrases: Optional[List[str]] = None,
        exception_phrases: Optional[List[str]] = None,
    ):
        self.topic_name = topic_name

        # Pre-compile word boundary patterns for identifiers and block words
        self.identifier_patterns: List[Tuple[str, re.Pattern]] = [
            (w.lower(), _word_boundary_pattern(w)) for w in identifier_words
        ]
        self.block_word_patterns: List[Tuple[str, re.Pattern]] = [
            (w.lower(), _word_boundary_pattern(w)) for w in block_words
        ]

        # Always-block phrases use substring matching (not word boundary)
        # because they're full phrases like "should i invest"
        self.always_block_phrases: List[str] = [
            p.lower() for p in (always_block_phrases or [])
        ]

        # Exception phrases use substring matching
        self.exception_phrases: List[str] = [
            p.lower() for p in (exception_phrases or [])
        ]


class TopicBlocker:
    """
    Simple topic blocker.

    Usage:
        blocker = TopicBlocker(denied_topics=[
            DeniedTopic(
                topic_name="investment",
                identifier_words=["stock", "invest", "crypto"],
                block_words=["buy", "sell", "price", "recommend"],
                always_block_phrases=["should i invest", "financial advice"],
                exception_phrases=["in stock", "invest time"],
            ),
        ])
        blocker.check("Should I buy stocks?")  # raises HTTPException 403
        blocker.check("Hello, how are you?")   # returns "Hello, how are you?"
    """

    def __init__(self, denied_topics: List[DeniedTopic]):
        self.denied_topics = denied_topics

    def check(self, text: str) -> str:
        """
        Check text against all denied topics.

        Returns the original text if allowed.
        Raises HTTPException(403) if blocked.
        """
        if not text or not text.strip():
            return text

        normalized = _normalize_text(text)

        # Check always-block phrases first (not affected by exceptions)
        for topic in self.denied_topics:
            for phrase in topic.always_block_phrases:
                if phrase in normalized:
                    verbose_proxy_logger.warning(
                        f"Topic blocker: always-block phrase '{phrase}' matched "
                        f"for topic '{topic.topic_name}'"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": f"Content blocked: denied topic '{topic.topic_name}'",
                            "topic": topic.topic_name,
                            "matched_phrase": phrase,
                            "match_type": "always_block",
                        },
                    )

        # Split into sentences for conditional matching
        sentences = _SENTENCE_SPLIT.split(normalized)

        for topic in self.denied_topics:
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if this sentence has an exception
                exception_found = False
                for exception in topic.exception_phrases:
                    if exception in sentence:
                        exception_found = True
                        break
                if exception_found:
                    continue

                # Check for identifier + block word in this sentence
                identifier_hit = None
                for word, pattern in topic.identifier_patterns:
                    if pattern.search(sentence):
                        identifier_hit = word
                        break

                if not identifier_hit:
                    continue

                block_word_hit = None
                for word, pattern in topic.block_word_patterns:
                    if pattern.search(sentence):
                        block_word_hit = word
                        break

                if block_word_hit:
                    matched = f"{identifier_hit} + {block_word_hit}"
                    verbose_proxy_logger.warning(
                        f"Topic blocker: conditional match '{matched}' "
                        f"for topic '{topic.topic_name}'"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": f"Content blocked: denied topic '{topic.topic_name}'",
                            "topic": topic.topic_name,
                            "matched_phrase": matched,
                            "match_type": "conditional",
                        },
                    )

        return text
