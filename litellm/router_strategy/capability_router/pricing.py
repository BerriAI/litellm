"""Request-cost estimates used only by capability selection."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.router_strategy.savings_baseline import canonical_model
from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.router import Router


@dataclass(frozen=True)
class _PricedModel:
    model: str
    deployment_id: str | None = None


def _deployment_model(deployment: Mapping[str, object]) -> _PricedModel | None:
    params = deployment.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    info_value = deployment.get("model_info")
    info = info_value if isinstance(info_value, Mapping) else {}
    model = info.get("base_model") or params.get("base_model") or params.get("model")
    provider_value = params.get("custom_llm_provider")
    provider = provider_value if isinstance(provider_value, str) else None
    qualified = canonical_model(model, provider) if isinstance(model, str) else None
    if qualified is None:
        return None
    deployment_id = info.get("id")
    return _PricedModel(qualified, str(deployment_id) if deployment_id else None)


def _models_served_by(router: "Router", model_group: str) -> tuple[_PricedModel, ...]:
    deployments = tuple(router.get_model_list(model_name=model_group) or ())
    if not deployments:
        direct = canonical_model(model_group)
        return (_PricedModel(direct),) if direct is not None else ()
    candidates = tuple(
        candidate
        for deployment in deployments
        if (candidate := _deployment_model(deployment)) is not None
    )
    return candidates if len(candidates) == len(deployments) else ()


def _request_cost(router: "Router", candidate: _PricedModel, usage: Usage) -> float | None:
    provider, _, model_name = candidate.model.partition("/")
    try:
        model_info = router.get_deployment_model_info(candidate.deployment_id or "", candidate.model)
        if model_info is None:
            return None
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model_name or candidate.model,
            usage=usage,
            custom_llm_provider=provider,
            model_info=model_info,
        )
    except Exception as exc:  # noqa: BLE001 - an unpriceable candidate uses the configured fallback
        verbose_router_logger.debug("CapabilityRouter: no pricing for %s (%s)", candidate.model, exc)
        return None
    cost = prompt_cost + completion_cost
    return cost if math.isfinite(cost) and cost >= 0 else None


def estimate_model_group_cost(router: "Router", model_group: str, usage: Usage) -> float | None:
    """Return a conservative cost estimate for every deployment behind a group."""
    candidates = _models_served_by(router, model_group)
    if not candidates:
        return None
    costs = tuple(_request_cost(router, candidate, usage) for candidate in candidates)
    if any(cost is None for cost in costs):
        return None
    return max(cost for cost in costs if cost is not None)
