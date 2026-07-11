"""
Declarative fallback generalizations for unknown / newly-released models.

The ``fallback_generalizations`` block in ``model_prices_and_context_window.json``
holds an ordered list of rules. Each rule pairs a single case-insensitive regex
with the metadata to apply when a model name has no exact entry in the cost map.
The metadata is a partial cost-map entry: ``litellm_provider`` drives provider
routing, and the remaining fields (``mode``, ``supports_*``, context window,
pricing, ...) drive ``get_model_info`` / ``supports_*``.

Precedence: rules are evaluated in file order and the first match wins. Callers
with extra constraints (model-info resolution checks the provider) use
``match_all_fallback_generalizations`` to skip inapplicable earlier rules instead
of discarding the model name. Rules are consulted only after exact and
case-insensitive lookups miss, so an exact entry always takes precedence over a
rule.

Patterns are matched case-insensitively with ``re.search`` and are not implicitly
anchored: a rule must include ``^`` and ``$`` (as the shipped rules do) to bind to
the whole model name, otherwise it matches as a substring. Keeping anchoring in the
regex makes the rule the single, self-contained source of truth for what it matches.

A rule may set ``extends`` to the ``name`` of another rule to inherit that rule's
``model_info``; the rule's own ``model_info`` overrides the inherited keys, so a
narrow rule (for example a version-gated capability flag) carries only its delta
instead of duplicating the parent's pricing block. Inheritance is resolved once,
at install time, against each rule's raw (unresolved) ``model_info``; it is a
single level (a parent that itself extends is not chained).

Any other keys on a rule (for example a free-text ``description`` documenting what
the regex matches) are ignored by the engine and exist only for the reader.

The compiled-regex list is built once and cached. ``match_fallback_generalization``
is O(number of rules); callers must only invoke it on a cache miss.
"""

import re
from typing import Optional

from litellm._logging import verbose_logger

NAME_FIELD = "name"
PATTERN_FIELD = "pattern"
MODEL_INFO_FIELD = "model_info"
EXTENDS_FIELD = "extends"


def _resolve_extends(rules: list) -> list:
    """Expand ``extends`` inheritance so each rule's ``model_info`` is self-contained.

    A rule with ``extends: <name>`` is rewritten with ``model_info`` set to the parent's
    ``model_info`` overlaid by its own. Resolution is single-level and uses each rule's
    raw ``model_info`` as the parent source. Non-dict rules and dangling parents are
    passed through unchanged.
    """
    base_by_name = {
        rule[NAME_FIELD]: rule[MODEL_INFO_FIELD]
        for rule in rules
        if isinstance(rule, dict)
        and isinstance(rule.get(NAME_FIELD), str)
        and isinstance(rule.get(MODEL_INFO_FIELD), dict)
    }

    def resolved(rule: dict) -> dict:
        parent_name = rule.get(EXTENDS_FIELD)
        own_info = rule.get(MODEL_INFO_FIELD)
        parent_info = base_by_name.get(parent_name) if isinstance(parent_name, str) else None
        if parent_info is None or not isinstance(own_info, dict):
            return rule
        return {**rule, MODEL_INFO_FIELD: {**parent_info, **own_info}}

    return [resolved(rule) if isinstance(rule, dict) else rule for rule in rules]


class _FallbackGeneralizations:
    """Holds the active rule list and its lazily-compiled regex cache."""

    def __init__(self) -> None:
        self.rules: list[dict] = []
        self._compiled: Optional[list[tuple[re.Pattern, dict]]] = None

    def set_rules(self, rules: Optional[list[dict]]) -> None:
        self.rules = rules if isinstance(rules, list) else []
        self._compiled = None

    def _compile(self) -> list[tuple[re.Pattern, dict]]:
        compiled: list[tuple[re.Pattern, dict]] = []
        for rule in self.rules:
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

    def matches(self, model: str) -> list[dict]:
        if not model:
            return []
        if self._compiled is None:
            self._compiled = self._compile()
        return [dict(model_info) for pattern, model_info in self._compiled if pattern.search(model) is not None]

    def match(self, model: str) -> Optional[dict]:
        return next(iter(self.matches(model)), None)


_registry = _FallbackGeneralizations()


def set_fallback_generalizations(rules: Optional[list[dict]]) -> None:
    """Install the active rule list and invalidate the compiled-regex cache.

    ``extends`` inheritance is resolved here, once, before the rules are stored.
    Called once when the model cost map is loaded (and again on any reload).
    """
    _registry.set_rules(_resolve_extends(rules) if isinstance(rules, list) else rules)


def get_fallback_generalization_rules() -> list[dict]:
    """Return the raw rule list (read-only view for callers/tests)."""
    return _registry.rules


def match_fallback_generalization(model: str) -> Optional[dict]:
    """Return the ``model_info`` of the first rule whose regex matches ``model``.

    O(number of rules). Only call this once exact lookups have missed.
    """
    return _registry.match(model)


def match_all_fallback_generalizations(model: str) -> list[dict]:
    """Return the ``model_info`` of every rule whose regex matches ``model``, in rule order.

    Lets a caller with extra constraints (e.g. a provider match) skip an
    inapplicable earlier rule instead of discarding the whole candidate.
    """
    return _registry.matches(model)
