from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import unquote_plus

from fastapi import status
from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLMRoutes, ProxyErrorTypes, ProxyException
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.types.proxy.request_tag_mappings import RequestTagMapping

REQUEST_TAG_MAPPINGS_ADAPTER: Final = TypeAdapter(tuple[RequestTagMapping, ...])


def is_baggage_mapped_route(route: str) -> bool:
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    return (
        (RouteChecks.is_llm_api_route(route) or InitPassThroughEndpointHelpers.is_registered_pass_through_route(route))
        and not RouteChecks.is_unbilled_route(route)
        and not RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.mcp_inference_routes.value)
    )


def mapping_rejection(message: str) -> ProxyException:
    return ProxyException(
        message=message, type=ProxyErrorTypes.bad_request_error, param=None, code=status.HTTP_400_BAD_REQUEST
    )


def resolve_baggage_tag_mappings(header: str, rules: Sequence[RequestTagMapping]) -> tuple[tuple[str, bool], ...]:
    """Exact mappings win over the any-value fallback for the same baggage key."""
    members: Final = tuple(_member_without_properties(member) for member in header.split(","))
    context: Final = W3CBaggagePropagator().extract({"baggage": ",".join(members)}, context=Context())
    values: Final = baggage.get_all(context)
    member_keys: Final = tuple(unquote_plus(member.partition("=")[0]).strip() for member in members)
    resolved: Final = (
        _resolve_key(key, values, member_keys, rules) for key in dict.fromkeys(rule.key for rule in rules)
    )
    return tuple(mapping for mapping in resolved if mapping is not None)


def _member_without_properties(member: str) -> str:
    key, separator, value = member.partition(";")[0].partition("=")
    return f"{key.strip()}{separator}{value.strip()}"


def _resolve_key(
    key: str, values: Mapping[str, object], member_keys: Sequence[str], rules: Sequence[RequestTagMapping]
) -> tuple[str, bool] | None:
    key_rules: Final = tuple(rule for rule in rules if rule.key == key)
    key_spellings: Final = frozenset((key, unquote_plus(key)))
    sent_keys: Final = tuple(member_key for member_key in member_keys if member_key in key_spellings)
    value: Final = values.get(sent_keys[0]) if len(sent_keys) == 1 else None
    selected: Final = next(
        (rule for rule in key_rules if rule.value == value and rule.value is not None), None
    ) or next((rule for rule in key_rules if rule.value is None), None)
    usable: Final = isinstance(value, str) and value.isprintable() and bool(value)
    if not usable or selected is None:
        if any(rule.required for rule in key_rules):
            raise mapping_rejection(f"Missing, invalid or unmapped baggage value for {key}")
        return None
    tag: Final = selected.tag if selected.tag is not None else f"{selected.tag_prefix}{value}"
    return tag, selected.require_budgeted_tag
