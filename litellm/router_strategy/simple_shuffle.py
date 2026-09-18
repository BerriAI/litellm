"""Choose among eligible deployments using request weights, then global metrics."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Final, TypeVar

from litellm.types.router_weights import validate_router_weights

_DeploymentT = TypeVar("_DeploymentT", bound=Mapping[str, object])
_ROUTER_LOGGER: Final = logging.getLogger("LiteLLM Router")


def _metric_weight(deployment: Mapping[str, object], metric: str) -> float:
    params: Final = deployment.get("litellm_params")
    value: Final = params.get(metric) if isinstance(params, Mapping) else None
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Deployment {metric} must be numeric")


def _scoped_weights(
    deployments: Sequence[Mapping[str, object]],
    model: str,
    request_kwargs: Mapping[str, object] | None,
) -> tuple[float, ...]:
    settings: Final = validate_router_weights((request_kwargs or {}).get("_router_weights"))
    model_weights: Final = settings.get(model) if settings is not None else None
    if not model_weights:
        return ()
    return tuple(
        model_weights.get(str(info.get("id")), 0.0) if isinstance(info, Mapping) else 0.0
        for deployment in deployments
        for info in (deployment.get("model_info"),)
    )


def simple_shuffle(
    resolve_model_alias: Callable[[str], str | None],
    healthy_deployments: Sequence[_DeploymentT],
    model: str,
    request_kwargs: Mapping[str, object] | None,
) -> _DeploymentT:
    resolved_model: Final = resolve_model_alias(model) or model
    weight_sets: Final = chain(
        (_scoped_weights(healthy_deployments, resolved_model, request_kwargs),),
        (
            tuple(_metric_weight(deployment, metric) for deployment in healthy_deployments)
            for metric in ("weight", "rpm", "tpm")
        ),
    )
    for weights in weight_sets:
        largest = max(weights, default=0.0)
        if largest <= 0:
            continue
        normalized = tuple(weight / largest for weight in weights)
        if sum(normalized) <= 0:
            continue
        selected = random.choices(healthy_deployments, weights=normalized)[0]
        _ROUTER_LOGGER.info("Selected deployment for model %s: %s", model, selected.get("model_info"))
        return selected
    return random.choice(healthy_deployments)
