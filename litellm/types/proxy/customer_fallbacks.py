from collections.abc import Mapping, Sequence
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Final, TypeAlias

from pydantic import AfterValidator

from litellm.models.fallback import FallbackRule


def _validated_rule(rule: Mapping[str, Sequence[str]]) -> tuple[str, Sequence[str]]:
    if len(rule) != 1:
        raise ValueError("Each customer fallback rule must contain exactly one requested model group")
    primary, targets = next(iter(rule.items()))
    if not primary.strip() or primary != primary.strip():
        raise ValueError("Requested model group names must be non-empty without surrounding whitespace")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("A customer fallback chain must contain at least one unique target model group")
    if any(not target.strip() or target != target.strip() for target in targets):
        raise ValueError("Fallback model group names must be non-empty without surrounding whitespace")
    return primary, targets


def _validate_customer_fallbacks(rules: Sequence[FallbackRule]) -> Sequence[FallbackRule]:
    entries: Final = tuple(_validated_rule(rule) for rule in rules)
    graph: Final[Mapping[str, Sequence[str]]] = dict(entries)
    if len(graph) != len(rules):
        raise ValueError("Each requested model group may have only one customer fallback rule")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError("Customer fallback rules must not contain cycles") from error
    return rules


CustomerFallbacks: TypeAlias = Annotated[list[FallbackRule], AfterValidator(_validate_customer_fallbacks)]
