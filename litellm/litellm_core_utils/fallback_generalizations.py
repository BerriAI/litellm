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

A rule that mixes ``litellm_provider`` with other ``model_info`` keys is invalid:
``set_fallback_generalizations`` logs a warning and skips it at install time (a
warning rather than a crash, because released proxies fetch this JSON remotely).
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


@dataclass(frozen=True, slots=True)
class _RoutingRule:
    pattern: re.Pattern
    provider: str


@dataclass(frozen=True, slots=True)
class _CapabilityRule:
    pattern: re.Pattern
    model_info: dict


_CompiledRule = Union[_RoutingRule, _CapabilityRule]


def _compile_rule(rule: object) -> Optional[_CompiledRule]:
    if not isinstance(rule, dict):
        return None
    pattern = rule.get(PATTERN_FIELD)
    model_info = rule.get(MODEL_INFO_FIELD)
    if not isinstance(pattern, str) or not isinstance(model_info, dict):
        verbose_logger.warning(
            "LiteLLM: skipping malformed fallback generalization rule %s (needs string '%s' and dict '%s').",
            rule.get(NAME_FIELD, pattern),
            PATTERN_FIELD,
            MODEL_INFO_FIELD,
        )
        return None
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        verbose_logger.warning(
            "LiteLLM: skipping fallback generalization rule with invalid regex %r: %s",
            pattern,
            e,
        )
        return None
    if PROVIDER_KEY not in model_info:
        return _CapabilityRule(pattern=compiled, model_info=model_info)
    provider = model_info[PROVIDER_KEY]
    if len(model_info) == 1 and isinstance(provider, str):
        return _RoutingRule(pattern=compiled, provider=provider)
    verbose_logger.warning(
        "LiteLLM: skipping invalid fallback generalization rule %s: a routing rule's '%s' must contain "
        "'%s' as its only key (a string), and a capability rule must not contain '%s' at all.",
        rule.get(NAME_FIELD, pattern),
        MODEL_INFO_FIELD,
        PROVIDER_KEY,
        PROVIDER_KEY,
    )
    return None


class _FallbackGeneralizations:
    """Holds the raw rule list and its install-time-compiled routing and capability rules."""

    def __init__(self) -> None:
        self.rules: list = []
        self.routing_rules: tuple = ()
        self.capability_rules: tuple = ()

    def set_rules(self, rules: Optional[list]) -> None:
        installed = rules if isinstance(rules, list) else []
        compiled = tuple(c for c in (_compile_rule(rule) for rule in installed) if c is not None)
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

    Malformed, invalid-regex, and mixed-kind rules are warned about and skipped here.
    Called once when the model cost map is loaded (and again on any reload).
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
