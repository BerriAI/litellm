"""
Declarative fallback generalizations for unknown / newly-released models.

The ``fallback_generalizations`` block in ``model_prices_and_context_window.json``
holds an ordered list of rules. Each rule pairs a single case-insensitive regex
with a ``model_info`` dict, and the structure of ``model_info`` decides which of
two kinds the rule is.

A ROUTING rule carries exactly one ``model_info`` key, ``litellm_provider``. It is
consumed only by ``get_llm_provider`` bare-id inference: the first routing rule
whose regex matches decides the provider. Routing rules never contribute to model
info.

A CAPABILITY rule carries any ``model_info`` keys except ``litellm_provider``
(``mode``, ``supports_*``, context window, pricing, ...). It is consumed by
``get_model_info`` fallback resolution: the ``model_info`` of ALL capability rules
whose regex matches is unioned in file order, with later rules overriding earlier
ones on key conflicts, and the caller backfills ``litellm_provider`` with the
provider it requested. If no capability rule matches, model-info resolution misses
as if no rules existed.

LEGACY-SCHEMA SHIM (temporary, until the new-schema JSON reaches main): released
proxies fetch this JSON remotely from main, whose block still ships the old schema
where a rule mixes ``litellm_provider`` with capability keys and may inherit a
parent's ``model_info`` via ``extends``. Such a legacy rule is tolerated rather
than skipped: ``extends`` is resolved once at install time (single level, against
raw parents), and the resolved rule acts as BOTH kinds, a routing rule (its
``litellm_provider`` participates in first-hit inference) and a capability rule
(its full ``model_info``, provider included, participates in the union). New-schema
rules never mix the two and never use ``extends``. A rule whose
``litellm_provider`` is not a string is invalid and is warned about and skipped
(a warning rather than a crash, for the same remote-fetch reason).

Rules are only consulted after exact and case-insensitive lookups miss, so an
exact cost-map entry always takes precedence over any rule.

Patterns are matched case-insensitively with ``re.search`` and are not implicitly
anchored: a rule must include ``^`` and ``$`` to bind to the whole model name,
otherwise it matches as a substring. Keeping anchoring in the regex makes the rule
the single, self-contained source of truth for what it matches.

Any other keys on a rule (for example a free-text ``description`` documenting what
the regex matches) are ignored by the engine and exist only for the reader.

Rules are compiled and classified once, at install time. The match functions are
O(number of rules); callers must only invoke them on a cache miss.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

from litellm._logging import verbose_logger

NAME_FIELD = "name"
PATTERN_FIELD = "pattern"
MODEL_INFO_FIELD = "model_info"
PROVIDER_KEY = "litellm_provider"
LEGACY_EXTENDS_FIELD = "extends"


def _resolve_legacy_extends(rules: list) -> list:
    """Expand legacy ``extends`` inheritance so each rule's ``model_info`` is self-contained.

    Compatibility shim for the old remote schema: single level, resolved against each
    parent's raw ``model_info``, with the child's own keys winning on conflict. Non-dict
    rules and dangling parents pass through unchanged; new-schema rules carry no
    ``extends`` and are untouched.
    """
    base_by_name = {
        rule[NAME_FIELD]: rule[MODEL_INFO_FIELD]
        for rule in rules
        if isinstance(rule, dict)
        and isinstance(rule.get(NAME_FIELD), str)
        and isinstance(rule.get(MODEL_INFO_FIELD), dict)
    }

    def resolved(rule: object) -> object:
        if not isinstance(rule, dict):
            return rule
        parent_name = rule.get(LEGACY_EXTENDS_FIELD)
        own_info = rule.get(MODEL_INFO_FIELD)
        parent_info = base_by_name.get(parent_name) if isinstance(parent_name, str) else None
        if parent_info is None or not isinstance(own_info, dict):
            return rule
        return {**rule, MODEL_INFO_FIELD: {**parent_info, **own_info}}

    return [resolved(rule) for rule in rules]


@dataclass(frozen=True, slots=True)
class _RoutingRule:
    pattern: re.Pattern
    provider: str


@dataclass(frozen=True, slots=True)
class _CapabilityRule:
    pattern: re.Pattern
    model_info: dict


_CompiledRule = Union[_RoutingRule, _CapabilityRule]


def _compile_rule(rule: object) -> tuple[_CompiledRule, ...]:
    if not isinstance(rule, dict):
        return ()
    pattern = rule.get(PATTERN_FIELD)
    model_info = rule.get(MODEL_INFO_FIELD)
    if not isinstance(pattern, str) or not isinstance(model_info, dict):
        verbose_logger.warning(
            "LiteLLM: skipping malformed fallback generalization rule %s (needs string '%s' and dict '%s').",
            rule.get(NAME_FIELD, pattern),
            PATTERN_FIELD,
            MODEL_INFO_FIELD,
        )
        return ()
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        verbose_logger.warning(
            "LiteLLM: skipping fallback generalization rule with invalid regex %r: %s",
            pattern,
            e,
        )
        return ()
    if PROVIDER_KEY not in model_info:
        return (_CapabilityRule(pattern=compiled, model_info=model_info),)
    provider = model_info[PROVIDER_KEY]
    if not isinstance(provider, str):
        verbose_logger.warning(
            "LiteLLM: skipping invalid fallback generalization rule %s: '%s' in '%s' must be a string.",
            rule.get(NAME_FIELD, pattern),
            PROVIDER_KEY,
            MODEL_INFO_FIELD,
        )
        return ()
    if len(model_info) == 1:
        return (_RoutingRule(pattern=compiled, provider=provider),)
    return (
        _RoutingRule(pattern=compiled, provider=provider),
        _CapabilityRule(pattern=compiled, model_info=model_info),
    )


class _FallbackGeneralizations:
    """Holds the raw rule list and its install-time-compiled routing and capability rules."""

    def __init__(self) -> None:
        self.rules: list = []
        self.routing_rules: tuple = ()
        self.capability_rules: tuple = ()

    def set_rules(self, rules: Optional[list]) -> None:
        installed = rules if isinstance(rules, list) else []
        compiled = tuple(kind for rule in _resolve_legacy_extends(installed) for kind in _compile_rule(rule))
        self.rules = installed
        self.routing_rules = tuple(rule for rule in compiled if isinstance(rule, _RoutingRule))
        self.capability_rules = tuple(rule for rule in compiled if isinstance(rule, _CapabilityRule))

    def match_routing(self, model: str) -> Optional[str]:
        if not model:
            return None
        return next(
            (rule.provider for rule in self.routing_rules if rule.pattern.search(model) is not None),
            None,
        )

    def match_capabilities(self, model: str) -> Optional[dict]:
        if not model:
            return None
        matched = tuple(rule.model_info for rule in self.capability_rules if rule.pattern.search(model) is not None)
        if not matched:
            return None
        return {key: value for model_info in matched for key, value in model_info.items()}


_registry = _FallbackGeneralizations()


def set_fallback_generalizations(rules: Optional[list]) -> None:
    """Install the active rule list, compiling and classifying each rule.

    Legacy ``extends`` inheritance is resolved here, once, before classification;
    a legacy rule mixing ``litellm_provider`` with capability keys installs as both
    kinds. Malformed and invalid-regex rules are warned about and skipped. Called
    once when the model cost map is loaded (and again on any reload).
    """
    _registry.set_rules(rules)


def get_fallback_generalization_rules() -> list:
    """Return the raw rule list (read-only view for callers/tests)."""
    return _registry.rules


def match_routing_generalization(model: str) -> Optional[str]:
    """Return the provider of the first routing rule whose regex matches ``model``.

    O(number of rules). Only call this once exact lookups have missed.
    """
    return _registry.match_routing(model)


def match_capability_generalizations(model: str) -> Optional[dict]:
    """Return the union of the ``model_info`` of every capability rule matching ``model``.

    Later rules override earlier ones on key conflicts. Returns ``None`` when no
    capability rule matches. O(number of rules); only call once exact lookups have missed.
    """
    return _registry.match_capabilities(model)
