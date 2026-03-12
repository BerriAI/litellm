"""
Tone checker: CPU-only regex detection of inappropriate chatbot tone.

Detects 6 categories: dismissive, blaming, refusal, condescension,
impatience, unprofessional.  Supports user-supplied blocked_phrases
(additional patterns) and safe_phrases (exemptions for domain jargon).
"""

import re
from typing import Dict, List, Optional, Pattern, Tuple

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Built-in tone patterns — each is (raw_regex, category_label)
# ---------------------------------------------------------------------------

_TONE_PATTERNS: List[Tuple[str, str]] = [
    # Dismissive
    (r"\bthat(?:'s| is) not (?:really )?my (?:problem|concern|issue)\b", "dismissive"),
    (r"\bi (?:don't|do not) see what the big deal is\b", "dismissive"),
    (r"\byou(?:'re| are) overthinking\b", "dismissive"),
    (r"\bjust read the (?:FAQ|docs|documentation|manual)\b", "dismissive"),

    # Blaming the customer
    # "you should have" is blame only when followed by a blame verb
    # (e.g. "you should have received" is informational, not blame)
    (r"\byou should have (?:read|known|checked|done|thought|realized|paid|looked|noticed|seen to)\b", "blaming"),
    (r"\bthat(?:'s| is) your (?:fault(?! tolerance| tolerant)|problem|mistake)\b", "blaming"),
    (r"\bif you had (?:followed|read|done)\b", "blaming"),
    (r"\byou clearly (?:didn't|did not)\b", "blaming"),
    (r"\bthis (?:issue|problem) is on your end\b", "blaming"),

    # Refusal to help (without offering alternatives)
    # "I can't help you" but NOT "I can't help but notice" (which is polite)
    (r"\bi (?:can't|cannot|can not) help you\b", "refusal"),
    # "nothing I can do" is refusal only when NOT followed by "but" / "to"
    (r"\bthere(?:'s| is) nothing (?:i|we) can do(?! (?:but|to))\b", "refusal"),
    (r"\byou(?:'ll| will) (?:just )?have to figure it out\b", "refusal"),
    (r"\btry somewhere else\b", "refusal"),

    # Sarcasm / condescension
    (r"\bif you(?:'d| had| would have) been paying attention\b", "condescension"),
    (r"\bhow (?:to )?make this any simpler\b", "condescension"),
    (r"\blet me spell it out for you\b", "condescension"),
    (r"\bsince you (?:don't|do not) (?:seem to )?get it\b", "condescension"),
    (r"\bdo your (?:job|work) for you\b", "condescension"),

    # Impatience / frustration — "told you" specifically, not "told our team"
    (r"\bi(?:'ve| have) already told you\b", "impatience"),
    (r"\bhow many times do i have to\b", "impatience"),
    (r"\bi (?:don't|do not) have time to\b", "impatience"),
    (r"\bare you even listening\b", "impatience"),
    (r"\bjust do what i said\b", "impatience"),

    # Unprofessional casual language
    (r"\b(?:bruh|lol|idk|smh|lmao|wtf)\b", "unprofessional"),
    (r"\bmy bad dude\b", "unprofessional"),
    (r"\bwhatever,? just deal with it\b", "unprofessional"),
    (r"\bsounds like a you problem\b", "unprofessional"),
]


def _compile_patterns(
    patterns: List[Tuple[str, str]],
) -> List[Tuple[Pattern[str], str]]:
    return [(re.compile(p, re.IGNORECASE), cat) for p, cat in patterns]


# Pre-compiled at module load
_COMPILED_TONE_PATTERNS = _compile_patterns(_TONE_PATTERNS)


class ToneChecker:
    """
    CPU-only tone checker.

    Config keys (all optional):
        blocked_phrases:  list of additional regex strings to block
        safe_phrases:     list of regex strings that exempt text from blocking
    """

    def __init__(self, config: Dict) -> None:
        self._extra_blocked: List[Tuple[Pattern[str], str]] = []
        for phrase in config.get("blocked_phrases") or []:
            self._extra_blocked.append(
                (re.compile(phrase, re.IGNORECASE), "custom_blocked")
            )

        self._safe_patterns: List[Pattern[str]] = []
        for phrase in config.get("safe_phrases") or []:
            self._safe_patterns.append(re.compile(phrase, re.IGNORECASE))

        verbose_proxy_logger.debug(
            "ToneChecker: initialized with %d extra blocked, %d safe phrases",
            len(self._extra_blocked),
            len(self._safe_patterns),
        )

    def _is_safe(self, text: str) -> bool:
        """Return True if text matches any user-defined safe phrase."""
        return any(p.search(text) for p in self._safe_patterns)

    def run(self, text: str) -> Optional[Dict]:
        """
        Check text for tone violations.

        Returns a dict with {matched_text, category} on first match, or None.
        """
        if self._is_safe(text):
            return None

        for pattern, category in _COMPILED_TONE_PATTERNS:
            m = pattern.search(text)
            if m:
                return {"matched_text": m.group(0), "category": category}

        for pattern, category in self._extra_blocked:
            m = pattern.search(text)
            if m:
                return {"matched_text": m.group(0), "category": category}

        return None
