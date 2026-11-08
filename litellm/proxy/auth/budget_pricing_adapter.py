from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._auth import budget_pricing
from litellm._auth.budget_pricing import DeploymentPrice, ModelPriceFacts, RouteBudgetFacts
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.decisions_pricing import decisions_pricing_fields, decisions_request_price
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.router import Router
from litellm.types.router import DeploymentTypedDict

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT: Final = TypeAdapter(object)


def _number(raw: object) -> float | None:
    return float(raw) if isinstance(raw, (float, int)) and not isinstance(raw, bool) else None


def _deployment_price(deployment: DeploymentTypedDict) -> DeploymentPrice:
    info: Final = _MAPPING.validate_python(deployment.get("model_info") or {})  # pyright: ignore[reportUnknownMemberType]  # deployment TypedDict contains legacy metadata
    model_id: Final = info.get("id")
    raw: Final = _OBJECT.validate_python(litellm.model_cost.get(model_id)) if model_id is not None else None  # pyright: ignore[reportUnknownMemberType]  # project one legacy global price entry
    entry: Final = as_str_mapping(raw) or {}
    params: Final = _MAPPING.validate_python(deployment["litellm_params"])
    provider: Final = params.get("custom_llm_provider") or str(params.get("model")).partition("/")[0]
    query_price: Final = decisions_request_price(
        {**entry, "litellm_provider": provider}, decisions_pricing_fields(entry)
    )
    return DeploymentPrice(
        has_input_price="input_cost_per_token" in entry,
        has_output_price="output_cost_per_token" in entry,
        input_price=_number(entry.get("input_cost_per_token")),
        output_price=_number(entry.get("output_cost_per_token")),
        query_price=query_price,
        ptu_count=0.0 if info.get("ptu_count") is not None else None,
        cost_per_ptu_per_hour=0.0 if info.get("cost_per_ptu_per_hour") is not None else None,
    )


def read_model_price_facts(model: str, router: Router, cached: bool | None = None) -> ModelPriceFacts:
    from litellm.proxy.auth.auth_checks import (
        _deployments_served_for,  # pyright: ignore[reportPrivateUsage]  # shared registry projection
    )

    if cached is not None:
        return ModelPriceFacts(cached=cached)
    try:
        group: Final = router.registry().group(model, include_hidden=True)
        facts: Final = ModelPriceFacts(
            group_found=group is not None,
            input_price=group.input_cost_per_token if group is not None else None,
            output_price=group.output_cost_per_token if group is not None else None,
        )
    except Exception:  # noqa: BLE001  # preserve lookup failure as non-exempt without caching
        return ModelPriceFacts(lookup_failed=True)
    if not budget_pricing.price(facts).read_deployments:
        return facts
    try:
        deployments: Final = tuple(
            _deployment_price(deployment) for deployment in _deployments_served_for(model, router)
        )
    except Exception:  # noqa: BLE001  # preserve lookup failure as non-exempt without caching
        return ModelPriceFacts(lookup_failed=True)
    return replace(facts, deployments=deployments)


def _read_cached_model_price(model: str, router: Router, cache: Mapping[str, bool] | None) -> ModelPriceFacts:
    cached: Final = cache.get(model) if isinstance(cache, dict) else None
    facts: Final = read_model_price_facts(model, router, cached)
    decision: Final = budget_pricing.price(facts)
    if isinstance(cache, dict) and decision.cache_value is not None:
        cache[model] = decision.cache_value  # rebind-ok: native-selected registry cache write
    return facts


def read_model_cost_facts(model: str, router: Router) -> ModelPriceFacts:
    return _read_cached_model_price(model, router, router.registry().zero_cost())


def is_model_cost_zero(model: str | Sequence[str] | None, llm_router: Router | None) -> bool:
    if model is None or llm_router is None:
        return budget_pricing.price(ModelPriceFacts()).zero_cost
    names: Final = (model,) if isinstance(model, str) else tuple(model)
    cache: Final = llm_router.registry().zero_cost()
    return all(budget_pricing.price(_read_cached_model_price(name, llm_router, cache)).zero_cost for name in names)


def read_route_budget_facts(route: str, body: Mapping[str, object] | None = None) -> RouteBudgetFacts:
    from litellm.proxy.auth.auth_checks import BUDGET_ENFORCED_SIDE_EFFECT_ROUTES, MODEL_DISCOVERY_ROUTES

    method: Final = body.get("method") if body is not None else None
    return RouteBudgetFacts(
        side_effect_route=route in BUDGET_ENFORCED_SIDE_EFFECT_ROUTES,
        discovery_route=route in MODEL_DISCOVERY_ROUTES,
        llm_route=RouteChecks.is_llm_api_route(route=route),
        route=route,
        method=method if isinstance(method, str) else None,
        body_present=body is not None,
    )


def _route_decision(route: str, body: Mapping[str, object] | None = None) -> budget_pricing.RouteBudgetDecision:
    facts: Final = read_route_budget_facts(route, body)
    decision: Final = budget_pricing.route(facts)
    if decision != "ReadPassThrough":
        return decision
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    registered: Final = InitPassThroughEndpointHelpers.is_registered_pass_through_route(route=route)
    return budget_pricing.route(replace(facts, registered_pass_through=registered))


def route_skips_budget_checks(route: str) -> bool:
    return _route_decision(route) == "Skip"


def read_budget_exemption_facts(
    route: str,
    model: str | Sequence[str] | None,
    llm_router: Router | None,
    *,
    request_body: Mapping[str, object] | None = None,
) -> budget_pricing.ExemptionFacts:
    names: Final = (model,) if isinstance(model, str) else tuple(model or ())
    initial: Final = budget_pricing.ExemptionFacts(
        read_route_budget_facts(route, request_body),
        model is not None,
        llm_router is not None,
        len(names),
    )
    return _read_exemption(initial, names, llm_router, None)


def _read_exemption(
    facts: budget_pricing.ExemptionFacts,
    models: tuple[str, ...],
    router: Router | None,
    cache: Mapping[str, bool] | None,
) -> budget_pricing.ExemptionFacts:
    current = facts  # rebind-ok: native-selected IO driver state
    selected_cache = cache  # rebind-ok: cache handle acquired at the first model read
    while True:
        decision = budget_pricing.exemption(current)
        if decision.read_pass_through:
            from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

            registered = InitPassThroughEndpointHelpers.is_registered_pass_through_route(route=current.route.route)
            current = replace(current, route=replace(current.route, registered_pass_through=registered))
            continue
        if decision.read_price_cache:
            assert router is not None
            selected_cache = router.registry().zero_cost()
            current = replace(current, cache_read=True)
            continue
        if decision.read_model is None:
            return current
        assert router is not None
        price = _read_cached_model_price(models[decision.read_model], router, selected_cache)
        current = replace(current, prices=(*current.prices, price))


def request_skips_budget_checks(
    route: str,
    model: str | Sequence[str] | None,
    llm_router: Router | None,
    *,
    request_body: Mapping[str, object] | None = None,
) -> bool:
    facts: Final = read_budget_exemption_facts(route, model, llm_router, request_body=request_body)
    return budget_pricing.exemption(facts).exempt
