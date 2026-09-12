import re
from typing import Final

from rapidfuzz import fuzz, process

from litellm.types.memory_v2 import MemoryEntry

_STOP_WORDS: Final = frozenset(
    "the and for how what why with this that does have from about our are was when should can you work team".split()
)
_TOKEN: Final = re.compile(r"[\w-]{2,}", re.UNICODE)
_PRIVATE_KEY: Final = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)
_CREDENTIAL: Final = re.compile(r"\b(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}")
_BEARER: Final = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE)


def redact_memory(value: str) -> str:
    return _BEARER.sub(
        r"\1[REDACTED]", _CREDENTIAL.sub("[REDACTED TOKEN]", _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value))
    )


def _similarity(term: str, text: str, words: tuple[str, ...]) -> float:
    if term in text:
        return 1.0
    match: Final = process.extractOne(term, words, scorer=fuzz.ratio, score_cutoff=66)
    return match[1] / 100 if match is not None else 0.0


def fuzzy_memories(
    query: str, entries: tuple[MemoryEntry, ...]
) -> tuple[tuple[MemoryEntry, float, tuple[str, ...]], ...]:
    if not query.strip():
        return tuple((entry, 0.0, ()) for entry in entries)
    tokens: Final = tuple(dict.fromkeys(match.group() for match in _TOKEN.finditer(query.casefold())))
    terms: Final = tuple(token for token in tokens if token not in _STOP_WORDS) or tokens

    def rank(entry: MemoryEntry) -> tuple[MemoryEntry, float, tuple[str, ...]]:
        fields: Final = (
            (entry.title.casefold(), 0.35),
            (entry.when_to_use.casefold(), 0.30),
            (entry.scope.casefold(), 0.20),
            (entry.content.casefold(), 0.15),
        )
        indexed: Final = tuple(
            (text, weight, tuple(frozenset(match.group() for match in _TOKEN.finditer(text))))
            for text, weight in fields
        )
        scores: Final = tuple(
            (
                term,
                sum(
                    score * weight
                    for text, weight, words in indexed
                    if (score := _similarity(term, text, words)) >= 0.66
                ),
            )
            for term in terms
        )
        return entry, sum(score for _, score in scores), tuple(term for term, score in scores if score > 0)

    return tuple(
        sorted((result for entry in entries if (result := rank(entry))[1] > 0), key=lambda r: (-r[1], r[0].memory_id))
    )
