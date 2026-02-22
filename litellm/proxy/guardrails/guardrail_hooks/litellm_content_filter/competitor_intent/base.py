"""
Generic competitor intent checker: two entity sets and overridable disambiguation.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, cast

from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
    CompetitorActionHint,
    CompetitorIntentEvidenceEntry,
    CompetitorIntentResult,
    CompetitorIntentType,
)

ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
LEET = {"@": "a", "4": "a", "0": "o", "3": "e", "1": "i", "5": "s", "7": "t"}

OTHER_MEANING_DEFAULT_THRESHOLD = (
    0.65  # Below this → treat as non-competitor (safe default).
)


def normalize(text: str) -> str:
    """Lowercase, NFKC, strip zero-width, leetspeak, collapse spaces."""
    if not text or not isinstance(text, str):
        return ""
    t = ZERO_WIDTH.sub("", text)
    t = unicodedata.normalize("NFKC", t).lower().strip()
    for c, r in LEET.items():
        t = t.replace(c, r)
    return re.sub(r"\s+", " ", t)


def _word_boundary_match(text: str, token: str) -> bool:
    """True if token appears as a word in text."""
    return bool(re.search(r"\b" + re.escape(token) + r"\b", text))


def _count_signals(text: str, patterns: List[str]) -> int:
    """Count how many of the patterns appear in text."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def _compile_marker(pattern: Optional[str]) -> Optional[Pattern[str]]:
    """Compile optional regex string to a pattern."""
    if not pattern or not pattern.strip():
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def text_for_entity_matching(text: str) -> str:
    """Letters-only variant for entity matching (e.g. split punctuation)."""
    t = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", t).strip()


class BaseCompetitorIntentChecker:
    """
    Generic competitor intent checker with two entity sets. Ambiguous tokens
    (competitor + other-meaning, e.g. location) are classified by overridable
    _classify_ambiguous(). Base implementation: treat as non-competitor.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.brand_self: List[str] = [
            s.lower().strip() for s in (config.get("brand_self") or []) if s
        ]
        competitors: List[str] = [
            s.lower().strip() for s in (config.get("competitors") or []) if s
        ]
        aliases_map: Dict[str, List[str]] = config.get("competitor_aliases") or {}
        self.competitor_canonical: Dict[str, str] = {}
        self._competitor_tokens: Set[str] = set()
        for c in competitors:
            self._competitor_tokens.add(c)
            self.competitor_canonical[c] = c
            for a in aliases_map.get(c) or []:
                a = a.lower().strip()
                if a:
                    self._competitor_tokens.add(a)
                    self.competitor_canonical[a] = c

        other: List[str] = [
            s.lower().strip() for s in (config.get("locations") or []) if s
        ]
        self._other_meaning_tokens: Set[str] = set(other)
        self._ambiguous: Set[str] = self._competitor_tokens & self._other_meaning_tokens

        self.policy: Dict[str, str] = config.get("policy") or {}
        self.threshold_high = float(config.get("threshold_high", 0.70))
        self.threshold_medium = float(config.get("threshold_medium", 0.45))
        self.threshold_low = float(config.get("threshold_low", 0.30))
        self.reframe_message_template: Optional[str] = config.get(
            "reframe_message_template"
        )
        self.refuse_message_template: Optional[str] = config.get(
            "refuse_message_template"
        )
        self._comparison_words: List[str] = list(
            config.get("comparison_words")
            or [
                "better",
                "worse",
                "best",
                "vs",
                "versus",
                "compare",
                "alternative",
                "recommend",
                "ranked",
            ]
        )
        self._domain_words: List[str] = [
            s.lower().strip() for s in (config.get("domain_words") or []) if s
        ]

    def _classify_ambiguous(self, text: str, token: str) -> Tuple[str, float]:
        """
        Override in subclasses for industry-specific logic. Base: treat as non-competitor.
        """
        return "OTHER_MEANING", 0.5

    def _find_matches(self, text: str) -> List[Tuple[str, str, bool]]:
        """Find competitor matches; mark ambiguous (also in other-meaning set)."""
        normalized = normalize(text)
        found: List[Tuple[str, str, bool]] = []
        seen: Set[Tuple[str, str]] = set()
        for token in self._competitor_tokens:
            if not _word_boundary_match(normalized, token):
                continue
            canonical = self.competitor_canonical.get(token, token)
            key = (token, canonical)
            if key in seen:
                continue
            seen.add(key)
            is_ambig = token in self._ambiguous or token in self._other_meaning_tokens
            found.append((token, canonical, is_ambig))
        return found

    def run(self, text: str) -> CompetitorIntentResult:
        """Classify competitor intent; non-competitor when ambiguous or low confidence."""
        normalized = normalize(text)
        evidence: List[CompetitorIntentEvidenceEntry] = []
        entities: Dict[str, List[str]] = {
            "brand_self": [],
            "competitors": [],
            "category": [],
        }

        for b in self.brand_self:
            if _word_boundary_match(normalized, b):
                entities["brand_self"].append(b)
                evidence.append(
                    {"type": "entity", "key": "brand_self", "value": b, "match": b}
                )

        matches = self._find_matches(text)
        if not matches:
            has_comparison = any(
                re.search(r"\b" + re.escape(w) + r"\b", normalized)
                for w in self._comparison_words
            )
            has_domain = self._domain_words and any(
                re.search(r"\b" + re.escape(w) + r"\b", normalized)
                for w in self._domain_words
            )
            if has_comparison and has_domain:
                evidence.append(
                    {
                        "type": "signal",
                        "key": "category_ranking",
                        "match": "comparison + domain",
                    }
                )
                action_hint = cast(
                    CompetitorActionHint,
                    self.policy.get("category_ranking", "reframe"),
                )
                return {
                    "intent": "category_ranking",
                    "confidence": 0.65,
                    "entities": entities,
                    "signals": ["category_ranking"],
                    "action_hint": action_hint,
                    "evidence": evidence,
                }
            return {
                "intent": "other",
                "confidence": 0.0,
                "entities": entities,
                "signals": [],
                "action_hint": "allow",
                "evidence": evidence,
            }

        competitor_resolved: List[str] = []
        for token, canonical, _ in matches:
            label, conf = self._classify_ambiguous(normalized, token)
            if label == "OTHER_MEANING":
                evidence.append(
                    {"type": "signal", "key": "other_meaning", "match": token}
                )
                continue
            if label == "COMPETITOR":
                competitor_resolved.append(canonical)
                evidence.append(
                    {
                        "type": "entity",
                        "key": "competitor",
                        "value": canonical,
                        "match": token,
                    }
                )
                if conf < OTHER_MEANING_DEFAULT_THRESHOLD:
                    competitor_resolved.pop()
                    evidence.append(
                        {
                            "type": "signal",
                            "key": "other_meaning_default",
                            "match": f"confidence {conf:.2f}",
                        }
                    )
                    continue

        entities["competitors"] = list(dict.fromkeys(competitor_resolved))

        if not competitor_resolved:
            return {
                "intent": "other",
                "confidence": 0.0,
                "entities": entities,
                "signals": ["other_meaning_or_ambiguous"],
                "action_hint": "allow",
                "evidence": evidence,
            }

        has_comparison = any(
            re.search(r"\b" + re.escape(w) + r"\b", normalized)
            for w in self._comparison_words
        )
        if has_comparison:
            evidence.append(
                {"type": "signal", "key": "comparison", "match": "comparison language"}
            )
        confidence = 0.75 if has_comparison else 0.55
        if confidence >= self.threshold_high:
            intent = "competitor_comparison"
        elif confidence >= self.threshold_medium:
            intent = "possible_competitor_comparison"
        elif confidence >= self.threshold_low:
            intent = "log_only"
        else:
            intent = "other"

        resolved_action_hint: CompetitorActionHint = cast(
            CompetitorActionHint, self.policy.get(intent, "allow")
        )
        if intent == "log_only":
            resolved_action_hint = "log_only"
        if intent == "other":
            resolved_action_hint = "allow"

        return {
            "intent": cast(CompetitorIntentType, intent),
            "confidence": round(confidence, 2),
            "entities": entities,
            "signals": ["competitor_resolved"]
            + (["comparison"] if has_comparison else []),
            "action_hint": resolved_action_hint,
            "evidence": evidence,
        }
