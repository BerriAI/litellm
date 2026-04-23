"""
Rule-based classifier mapping a user prompt to a RequestType.

V0 design choice: deterministic regex over the FIRST user message in a session.
Result is cached per session (caller's responsibility, not ours).

Order matters: we check more specific types first, falling back to GENERAL.
"""

import re
from typing import List, Pattern, Tuple

from litellm.types.router import RequestType

_RULES: List[Tuple[Pattern[str], RequestType]] = [
    (
        re.compile(
            r"\b(write|create|generate|implement|build)\s+(?:a |an |the |me )?(?:python|javascript|typescript|java|rust|go|c\+\+|sql|bash|shell)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_GENERATION,
    ),
    (
        re.compile(
            r"\b(write|create|implement|build)\b(?:\s+\w+){0,4}?\s+(function|class|method|script|program|api|endpoint|microservice)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_GENERATION,
    ),
    (
        re.compile(
            r"\b(explain|describe|understand|walk me through|what does)\b.*\b(code|function|method|class|algorithm|snippet)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(debug|fix|why (?:is|does|isn't)|what.s wrong|trace)\b.*\b(error|bug|exception|stacktrace|stack trace|traceback)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(review|critique)\s+(?:this |my |the )?(?:code|pr|pull request|diff|patch)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(design|architect|plan|architecture)\b.*\b(system|service|api|database|schema|module|microservice)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\b(should i (?:use|choose|pick)|tradeoffs? between|compare)\b.*\b(library|framework|language|database|protocol|postgres|postgresql|mongodb|dynamodb|mysql|redis|kafka|sql|nosql)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\bhow (?:should|do) i (?:design|structure|organize|model)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\b(solve|compute|calculate|prove|derive)\b.*\b(equation|integral|derivative|theorem|proof|problem)\b",
            re.IGNORECASE,
        ),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(r"\b(if .+ then|given .+ find|suppose|assume)\b", re.IGNORECASE),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(
            r"\b(probability|statistics|combinatorics|optimization problem)\b",
            re.IGNORECASE,
        ),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(
            r"\b(write|draft|compose|rewrite|edit|proofread|polish)\b.*\b(email|essay|blog|post|article|letter|memo|copy|paragraph|sentence)\b",
            re.IGNORECASE,
        ),
        RequestType.WRITING,
    ),
    (
        re.compile(
            r"\b(make (?:this|it)|help me)\s+(?:more |less )?(?:concise|formal|casual|professional|persuasive)\b",
            re.IGNORECASE,
        ),
        RequestType.WRITING,
    ),
    (
        re.compile(
            r"^\s*(who|what|when|where|which)\s+(?:is|was|were|are)\b", re.IGNORECASE
        ),
        RequestType.FACTUAL_LOOKUP,
    ),
    (
        re.compile(r"^\s*(define|definition of|meaning of)\b", re.IGNORECASE),
        RequestType.FACTUAL_LOOKUP,
    ),
    (
        re.compile(
            r"^\s*how (?:do you spell|to spell|many .* are there|tall is)\b",
            re.IGNORECASE,
        ),
        RequestType.FACTUAL_LOOKUP,
    ),
]


def classify_prompt(text: str) -> RequestType:
    """
    Classify a single user prompt.

    Falls back to GENERAL when no rule matches. Empty/whitespace-only also
    returns GENERAL.
    """
    if not text or not text.strip():
        return RequestType.GENERAL

    truncated = text[:2000]

    for pattern, request_type in _RULES:
        if pattern.search(truncated):
            return request_type

    return RequestType.GENERAL
