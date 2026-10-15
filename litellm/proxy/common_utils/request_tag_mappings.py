from collections.abc import Mapping, Sequence
from typing import Final

from fastapi import HTTPException
from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context

from litellm.types.proxy.request_tag_mappings import RequestTagMapping


def resolve_baggage_tag_mappings(header: str, rules: Sequence[RequestTagMapping]) -> tuple[tuple[str, bool], ...]:
    """Exact mappings win over the any-value fallback for the same baggage key."""
    without_properties: Final = ",".join(member.partition(";")[0] for member in header.split(","))
    context: Final = W3CBaggagePropagator().extract({"baggage": without_properties}, context=Context())
    values: Final = baggage.get_all(context)
    member_keys: Final = tuple(member.partition("=")[0].strip() for member in header.split(","))
    resolved: Final = (
        _resolve_key(key, values, member_keys, rules) for key in dict.fromkeys(rule.key for rule in rules)
    )
    return tuple(mapping for mapping in resolved if mapping is not None)


def _resolve_key(
    key: str, values: Mapping[str, object], member_keys: Sequence[str], rules: Sequence[RequestTagMapping]
) -> tuple[str, bool] | None:
    key_rules: Final = tuple(rule for rule in rules if rule.key == key)
    value: Final = values.get(key)
    selected: Final = next(
        (rule for rule in key_rules if rule.value == value and rule.value is not None), None
    ) or next((rule for rule in key_rules if rule.value is None), None)
    usable: Final = isinstance(value, str) and bool(value) and member_keys.count(key) == 1
    if not usable or selected is None:
        if any(rule.required for rule in key_rules):
            raise HTTPException(400, f"Missing, invalid or unmapped baggage value for {key}")
        return None
    tag: Final = selected.tag if selected.tag is not None else f"{selected.tag_prefix}{value}"
    return tag, selected.require_budgeted_tag
