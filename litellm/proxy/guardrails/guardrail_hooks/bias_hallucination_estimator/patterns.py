from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern, Tuple


@dataclass(frozen=True, slots=True)
class PatternRule:
    name: str
    pattern: Pattern[str]
    score: float


BIAS_PATTERNS: Tuple[PatternRule, ...] = (
    PatternRule(
        name="dogmatic_language",
        pattern=re.compile(
            r"\b(?:always|never|obviously|clearly|undeniably|without question|"
            r"everyone knows|common sense|the fact is)\b",
            re.IGNORECASE,
        ),
        score=0.18,
    ),
    PatternRule(
        name="opinion_as_fact",
        pattern=re.compile(
            r"\b(?:i believe|i think|in my opinion|should be|must be|has to be|"
            r"need to be)\b",
            re.IGNORECASE,
        ),
        score=0.16,
    ),
    PatternRule(
        name="overconfidence",
        pattern=re.compile(
            r"\b(?:100%|guaranteed|certainly|definitely|there is no doubt|"
            r"cannot be wrong|will definitely)\b",
            re.IGNORECASE,
        ),
        score=0.22,
    ),
    PatternRule(
        name="sweeping_generalization",
        pattern=re.compile(
            r"\b(?:all|no|every)\s+[a-z][a-z\-]{2,}(?:\s+[a-z][a-z\-]{2,}){0,2}"
            r"\s+(?:are|is|can|cannot|can't|will|won't|should|must)\b",
            re.IGNORECASE,
        ),
        score=0.24,
    ),
)


UNSOURCED_STATISTIC_PATTERN = PatternRule(
    name="unsourced_statistics",
    pattern=re.compile(
        r"(?:\b\d+(?:\.\d+)?\s?%|\b\d+\s+(?:out of|in)\s+\d+\b|"
        r"\b\d+(?:,\d{3})+(?:\.\d+)?\b)",
        re.IGNORECASE,
    ),
    score=0.32,
)


CITATION_GAP_PATTERNS: Tuple[PatternRule, ...] = (
    PatternRule(
        name="unnamed_research",
        pattern=re.compile(
            r"\b(?:research shows|studies show|studies found|a study found|"
            r"scientists found|experts say|experts agree|according to experts)\b",
            re.IGNORECASE,
        ),
        score=0.3,
    ),
    PatternRule(
        name="vague_authority",
        pattern=re.compile(
            r"\b(?:it is widely known|it has been proven|data proves|"
            r"evidence proves)\b",
            re.IGNORECASE,
        ),
        score=0.26,
    ),
)


FABRICATED_SPECIFICITY_PATTERNS: Tuple[PatternRule, ...] = (
    PatternRule(
        name="overly_precise_number",
        pattern=re.compile(
            r"\b(?:exactly|precisely)\s+\d+(?:,\d{3})*(?:\.\d+)?\b|"
            r"\b\d+\.\d{3,}\b",
            re.IGNORECASE,
        ),
        score=0.24,
    ),
    PatternRule(
        name="specific_date_claim",
        pattern=re.compile(
            r"\bon\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
            r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
            re.IGNORECASE,
        ),
        score=0.18,
    ),
)


SOURCE_INDICATOR_PATTERN = re.compile(
    r"\b(?:according to|cited by|source:|doi:|isbn|https?://|www\.|"
    r"published in|published by|journal|report|whitepaper|survey by|study by|"
    r"dataset|citation|reference)\b",
    re.IGNORECASE,
)
