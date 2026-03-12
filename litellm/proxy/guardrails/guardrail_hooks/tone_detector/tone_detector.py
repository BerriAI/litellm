"""
Tone Detector Guardrail for LiteLLM.

CPU-only guardrail that detects inappropriate tone in customer-facing chatbot
responses (dismissive, condescending, blaming, unprofessional language) while
allowing domain-specific terms that might otherwise trigger false positives.
"""

import re
from typing import TYPE_CHECKING, List, Literal, Optional, Pattern, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


# ---------------------------------------------------------------------------
# Built-in tone patterns — each is (compiled_regex, category_label)
# ---------------------------------------------------------------------------

_TONE_PATTERNS: List[Tuple[str, str]] = [
    # Dismissive
    (r"\bthat(?:'s| is) not (?:really )?my (?:problem|concern|issue)\b", "dismissive"),
    (r"\bi (?:don't|do not) see what the big deal is\b", "dismissive"),
    (r"\byou(?:'re| are) overthinking\b", "dismissive"),
    (r"\bjust read the (?:FAQ|docs|documentation|manual)\b", "dismissive"),

    # Blaming the customer
    # "you should have" is blame only when NOT followed by a past participle
    # describing something the customer already received/seen
    # (e.g. "you should have received" is informational, not blame)
    (r"\byou should have (?:read|known|checked|done|thought|realized|paid|looked|noticed|seen to)\b", "blaming"),
    (r"\bthat(?:'s| is) your (?:fault(?! tolerance| tolerant)|problem|mistake)\b", "blaming"),
    (r"\bif you had (?:followed|read|done)\b", "blaming"),
    (r"\byou clearly (?:didn't|did not)\b", "blaming"),
    (r"\bthis (?:issue|problem) is on your end\b", "blaming"),

    # Refusal to help (without offering alternatives)
    # "I can't help you" but NOT "I can't help but notice" (which is polite)
    (r"\bi (?:can't|cannot|can not) help you\b", "refusal"),
    # "nothing I can do" is refusal only when NOT followed by "but" / "to" + helpful verb
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


class ToneDetectorGuardrail(CustomGuardrail):
    """
    Detects inappropriate tone in chatbot responses.

    Configuration accepts:
        blocked_phrases:  additional regex patterns to block
        safe_phrases:     regex patterns that exempt a text from blocking
                          (e.g. domain-specific jargon)
    """

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        blocked_phrases: Optional[List[str]] = None,
        safe_phrases: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ],
            **kwargs,
        )

        # User-supplied additional blocked patterns
        self._extra_blocked: List[Tuple[Pattern[str], str]] = []
        for phrase in blocked_phrases or []:
            self._extra_blocked.append(
                (re.compile(phrase, re.IGNORECASE), "custom_blocked")
            )

        # User-supplied safe-phrase patterns — if ANY safe phrase matches
        # the text, that text is allowed through even if a tone pattern fires.
        self._safe_patterns: List[Pattern[str]] = []
        for phrase in safe_phrases or []:
            self._safe_patterns.append(re.compile(phrase, re.IGNORECASE))

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def _is_safe(self, text: str) -> bool:
        """Return True if text matches any user-defined safe phrase."""
        return any(p.search(text) for p in self._safe_patterns)

    def _check_tone(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Check a single text for tone violations.

        Returns (matched_text, category) on first match, or None.
        """
        if self._is_safe(text):
            return None

        for pattern, category in _COMPILED_TONE_PATTERNS:
            m = pattern.search(text)
            if m:
                return (m.group(0), category)

        for pattern, category in self._extra_blocked:
            m = pattern.search(text)
            if m:
                return (m.group(0), category)

        return None

    # ------------------------------------------------------------------
    # CustomGuardrail hook
    # ------------------------------------------------------------------

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts") or []

        for text in texts:
            if not text:
                continue
            result = self._check_tone(text)
            if result is not None:
                matched, category = result
                verbose_proxy_logger.warning(
                    "ToneDetector blocked (%s): '%s'",
                    category,
                    matched,
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"Tone violation detected: {category}",
                        "category": category,
                        "matched_text": matched,
                    },
                )

        return inputs
