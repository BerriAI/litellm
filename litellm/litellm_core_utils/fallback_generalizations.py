"""
Declarative fallback generalizations for unknown / newly-released models.

The ``fallback_generalizations`` block in ``model_prices_and_context_window.json``
holds an ordered list of rules. Each rule pairs a single case-insensitive regex
with the metadata to apply when a model name has no exact entry in the cost map.
The metadata is a partial cost-map entry: ``litellm_provider`` drives provider
routing, and the remaining fields (``mode``, ``supports_*``, context window,
pricing, ...) drive ``get_model_info`` / ``supports_*``.

Precedence: rules are evaluated in file order and the first match wins. They are
consulted only after exact and case-insensitive lookups miss, so an exact entry
always takes precedence over a rule.

Patterns are matched case-insensitively with ``re.search`` and are not implicitly
anchored: a rule must include ``^`` and ``$`` (as the shipped rules do) to bind to
the whole model name, otherwise it matches as a substring. Keeping anchoring in the
regex makes the rule the single, self-contained source of truth for what it matches.

The compiled-regex list is built once and cached. ``match_fallback_generalization``
is O(number of rules); callers must only invoke it on a cache miss.
"""

import re
from typing import Dict, List, Optional, Pattern, Tuple

from litellm._logging import verbose_logger

PATTERN_FIELD = "pattern"
MODEL_INFO_FIELD = "model_info"

_rules: List[dict] = []
_compiled_rules: Optional[List[Tuple[Pattern, Dict]]] = None


def set_fallback_generalizations(rules: Optional[List[dict]]) -> None:
    """Install the active rule list and invalidate the compiled-regex cache.

    Called once when the model cost map is loaded (and again on any reload).
    """
    global _rules, _compiled_rules
    _rules = rules if isinstance(rules, list) else []
    _compiled_rules = None


def get_fallback_generalization_rules() -> List[dict]:
    """Return the raw rule list (read-only view for callers/tests)."""
    return _rules


def _compile_rules() -> List[Tuple[Pattern, Dict]]:
    compiled: List[Tuple[Pattern, Dict]] = []
    for rule in _rules:
        if not isinstance(rule, dict):
            continue
        pattern = rule.get(PATTERN_FIELD)
        model_info = rule.get(MODEL_INFO_FIELD)
        if not isinstance(pattern, str) or not isinstance(model_info, dict):
            verbose_logger.warning(
                "LiteLLM: skipping malformed fallback generalization rule %s (needs string '%s' and dict '%s').",
                rule.get("name", pattern),
                PATTERN_FIELD,
                MODEL_INFO_FIELD,
            )
            continue
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), model_info))
        except re.error as e:
            verbose_logger.warning(
                "LiteLLM: skipping fallback generalization rule with invalid regex %r: %s",
                pattern,
                e,
            )
    return compiled


def match_fallback_generalization(model: str) -> Optional[Dict]:
    """Return the ``model_info`` of the first rule whose regex matches ``model``.

    O(number of rules). Only call this once exact lookups have missed.
    """
    global _compiled_rules
    if not model:
        return None
    if _compiled_rules is None:
        _compiled_rules = _compile_rules()
    for pattern, model_info in _compiled_rules:
        if pattern.search(model) is not None:
            return dict(model_info)
    return None
