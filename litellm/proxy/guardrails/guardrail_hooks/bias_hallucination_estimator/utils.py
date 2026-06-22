from __future__ import annotations

import re
from typing import Iterable, Tuple

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> Tuple[str, ...]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return ()
    return tuple(
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(normalized_text)
        if sentence.strip()
    )


def clip_example(text: str, max_length: int = 160) -> str:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return f"{normalized_text[: max_length - 3]}..."


def unique_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
