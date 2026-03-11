"""
Airline-specific competitor intent: other meaning (e.g. location/travel context) vs competitor airline.

Uses context-based disambiguation only: no hardcoded place lists. Detects travel-location
language (prepositions, travel verbs, booking/entry nouns) vs airline context (airways,
carrier, lounge, miles, etc.) and scores to decide OTHER_MEANING vs COMPETITOR.

When competitors is not provided, loads major_airlines.json and excludes the customer's
brand_self so all other major airlines are treated as competitors.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent.base import (
    BaseCompetitorIntentChecker,
    _compile_marker,
    _count_signals,
    _word_boundary_match,
)

# Location/travel context: prepositions, travel verbs, booking nouns, entry/geo nouns.
# No place-name list; these patterns detect "destination context" generically.
AIRLINE_OTHER_MEANING_SIGNALS = [
    # Travel verb + preposition (e.g. "fly to", "layover in")
    r"\b(fly|flying|travel|traveling|going|visit|visiting|transit|layover|stopover)\b.{0,12}\b(to|from|via|in|at|through|into)\b",
    # Booking + preposition
    r"\bflight(s)?\b.{0,10}\b(to|from|via)\b",
    r"\bticket(s)?\b.{0,8}\b(to|for)\b",
    r"\bfare(s)?\b.{0,8}\b(to)\b",
    # Entry/geo/booking single words
    r"\bvisa\b",
    r"\bimmigration\b",
    r"\bcustoms\b",
    r"\bentry\b",
    r"\bairport\b",
    r"\bterminal\b",
    r"\bgate\b",
    r"\bdeparture\b",
    r"\barrival\b",
    r"\bitinerary\b",
    r"\bweather\b",
    r"\bhotel\b",
    r"\bcity\b",
    # Prepositions alone (weaker; often near a place)
    r"\bto\s+",
    r"\bfrom\s+",
    r"\bin\s+",
    r"\bat\s+",
    r"\bvia\s+",
]

# Airline context: carrier/airline language, cabin, loyalty, operations.
# If ambiguous token appears near these → treat as COMPETITOR.
AIRLINE_COMPETITOR_SIGNALS = [
    r"\bairways?\b",
    r"\bairline\b",
    r"\bcarrier\b",
    r"\bcabin\s+crew\b",
    r"\bflight\s+attendant\b",
    r"\bbusiness\s+class\b",
    r"\bfirst\s+class\b",
    r"\beconomy\b",
    r"\blounge\b",
    r"\bbaggage\s+allowance\b",
    r"\bcheck[- ]?in\b",
    r"\bmiles\b",
    r"\bloyalty\b",
    r"\bstatus\b",
    r"\bfrequent\s+flyer\b",
    r"\bfleet\b",
    r"\baircraft\b",
    # Comparison/ranking
    r"\bbetter\b",
    r"\bbest\b",
    r"\bgood\b",
    r"\bas\s+good\s+as\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
    r"\balternative\b",
    r"\bcompetitor\b",
    # Brand-specific (optional; config can extend)
    r"\bqmiles\b",
    r"\bprivilege\s+club\b",
]

# Operational-only: baggage, lounge, check-in, refund (no comparison language).
# When only these appear with ambiguous token → treat as product query (OTHER_MEANING).
AIRLINE_OPERATIONAL_SIGNALS = [
    r"\bbaggage\s+allowance\b",
    r"\blounge\b",
    r"\bcheck[- ]?in\b",
    r"\brefund\b",
    r"\bpremium\s+lounge\b",
]
# Comparison language: if present with competitor signals → COMPETITOR.
AIRLINE_COMPARISON_SIGNALS = [
    r"\bbetter\b",
    r"\bbest\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
]

# Explicit markers: strong override when present.
AIRLINE_EXPLICIT_COMPETITOR_MARKER = r"\b(airways?|airline|carrier)\b"
AIRLINE_EXPLICIT_OTHER_MEANING_MARKER = (
    r"\b(fly|travel|going|visit|layover|stopover|transit)\b.{0,12}\b(to|in|via|from)\b.{0,8}\b"
)

_MAJOR_AIRLINES_PATH = (
    Path(__file__).resolve().parent / "major_airlines.json"
)


def _load_competitors_excluding_brand(brand_self: List[str]) -> List[str]:
    """
    Load competitor tokens from major_airlines.json (harm_toxic_abuse-style format).
    Exclude any airline whose id or match variants overlap with brand_self.
    Returns a flat list of match variants (pipe-separated values) from non-excluded airlines.
    """
    brand_set = {b.lower().strip() for b in brand_self if b}
    if not _MAJOR_AIRLINES_PATH.exists():
        return []
    try:
        with open(_MAJOR_AIRLINES_PATH, encoding="utf-8") as f:
            airlines = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    result: List[str] = []
    for entry in airlines:
        if not isinstance(entry, dict):
            continue
        match_str = entry.get("match") or ""
        variants = [v.strip().lower() for v in match_str.split("|") if v.strip()]
        words_in_match: Set[str] = set()
        for v in variants:
            words_in_match.update(v.split())
        if brand_set & words_in_match or any(v in brand_set for v in variants):
            continue
        result.extend(variants)
    return result


class AirlineCompetitorIntentChecker(BaseCompetitorIntentChecker):
    """
    Disambiguates other meaning (e.g. country/city/airport) vs competitor airline
    (e.g. "Qatar" → country vs Qatar Airways). Overrides _classify_ambiguous
    with other_meaning/competitor signals and explicit markers.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        merged: Dict[str, Any] = dict(config)
        if not merged.get("other_meaning_signals"):
            merged["other_meaning_signals"] = AIRLINE_OTHER_MEANING_SIGNALS
        if not merged.get("competitor_signals"):
            merged["competitor_signals"] = AIRLINE_COMPETITOR_SIGNALS
        # Optional: no default place list; config can add other_meaning_anchors for extra patterns
        if "other_meaning_anchors" not in merged:
            merged["other_meaning_anchors"] = []
        if not merged.get("explicit_competitor_marker"):
            merged["explicit_competitor_marker"] = AIRLINE_EXPLICIT_COMPETITOR_MARKER
        if not merged.get("explicit_other_meaning_marker"):
            merged["explicit_other_meaning_marker"] = AIRLINE_EXPLICIT_OTHER_MEANING_MARKER
        if not merged.get("domain_words"):
            merged["domain_words"] = ["airline", "airlines", "carrier"]
        if not merged.get("competitors"):
            merged["competitors"] = _load_competitors_excluding_brand(
                merged.get("brand_self") or []
            )
        super().__init__(merged)
        self._other_meaning_signals = list(merged.get("other_meaning_signals") or [])
        self._competitor_signals = list(merged.get("competitor_signals") or [])
        self._other_meaning_anchors = list(merged.get("other_meaning_anchors") or [])
        self._explicit_competitor_marker = _compile_marker(
            merged.get("explicit_competitor_marker")
        )
        self._explicit_other_meaning_marker = _compile_marker(
            merged.get("explicit_other_meaning_marker")
        )

    def _classify_ambiguous(self, text: str, token: str) -> Tuple[str, float]:
        """Other meaning vs competitor using airline signals and explicit markers."""
        text_lower = text.lower()
        if self._explicit_competitor_marker and self._explicit_competitor_marker.search(
            text_lower
        ) and _word_boundary_match(text_lower, token.lower()):
            return "COMPETITOR", 0.85
        if self._explicit_other_meaning_marker and self._explicit_other_meaning_marker.search(
            text_lower
        ):
            return "OTHER_MEANING", 0.85
        # Operational-only: baggage/lounge/check-in/refund with no comparison → product query
        has_comparison = _count_signals(text_lower, AIRLINE_COMPARISON_SIGNALS) > 0
        operational_count = _count_signals(text_lower, AIRLINE_OPERATIONAL_SIGNALS)
        if not has_comparison and operational_count > 0:
            return "OTHER_MEANING", 0.85
        # Score: location/travel context vs airline context (no place-name list)
        other_count = _count_signals(text_lower, self._other_meaning_signals)
        if self._other_meaning_anchors:
            other_count += _count_signals(text_lower, self._other_meaning_anchors)
        comp_count = _count_signals(text_lower, self._competitor_signals)
        total = other_count + comp_count
        if total == 0:
            return "OTHER_MEANING", 0.5
        other_ratio = other_count / total
        comp_ratio = comp_count / total
        if other_ratio >= 0.6:
            return "OTHER_MEANING", min(0.9, 0.5 + 0.4 * other_ratio)
        if comp_ratio >= 0.6:
            return "COMPETITOR", min(0.9, 0.5 + 0.4 * comp_ratio)
        return "OTHER_MEANING", 0.5
