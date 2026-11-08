from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from pydantic import BaseModel, Field, TypeAdapter

import litellm
from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.unpriced_models import DeploymentPricing, PriceEntry, PriceMetric, UnpricedModelFacts, unpriced_model
from litellm.router_api import RegistryReader

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OPTIONAL_STRING: Final = TypeAdapter[str | None](str | None)


class _ModelInfo(BaseModel, frozen=True):
    id: str | None = None


class _Deployment(BaseModel, frozen=True):
    litellm_params: Mapping[str, object] = Field(default_factory=dict[str, object])
    model_info: _ModelInfo = Field(default_factory=_ModelInfo)


class _CostMap(BaseModel, frozen=True):
    model_cost: Mapping[str, object]


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metric(name: str, value: object) -> PriceMetric:
    return PriceMetric(
        name,
        _number(value),
        tuple(_number(item) for item in _MAPPING.validate_python(value).values()) if isinstance(value, dict) else (),
    )


def price_entry(entry: Mapping[str, object]) -> PriceEntry:
    return PriceEntry(
        entry.get("tiered_pricing") is not None, tuple(_metric(key, value) for key, value in entry.items())
    )


def read_unpriced_model_facts(model: str | None, llm_router: RegistryReader | None) -> UnpricedModelFacts:
    return _read(UnpricedModelFacts(model, llm_router is not None), llm_router, ())


def _read(facts: UnpricedModelFacts, router: RegistryReader | None, rows: Sequence[_Deployment]) -> UnpricedModelFacts:
    decision: Final = unpriced_model(facts)
    if decision.kind == "Complete":
        return facts
    if router is None or facts.model is None:
        raise NativeAuthRuntimeError("unpriced model read without a router or model")
    match decision.kind:
        case "ReadGroup":
            found: Final = router.registry().group(facts.model) is not None
            return _read(replace(facts, group_found=found), router, rows)
        case "ReadDeployments" | "ReadCostDeployments":
            deployments: Final = tuple(
                _Deployment.model_validate(row) for row in router.registry().deployments(facts.model) or ()
            )
            projected: Final = tuple(
                DeploymentPricing(tuple(row.litellm_params), row.model_info.id is not None) for row in deployments
            )
            snapshot: Final = (
                replace(facts, deployments=projected)
                if decision.kind == "ReadDeployments"
                else replace(facts, cost_deployments=projected)
            )
            return _read(snapshot, router, deployments)
        case "ReadInfo" | "ReadCost":
            if facts.deployments is None:
                raise NativeAuthRuntimeError("missing unpriced deployment facts")
            row: Final = rows[decision.index]
            model_id: Final = row.model_info.id
            if model_id is None:
                raise NativeAuthRuntimeError("missing unpriced deployment identifier")
            current: Final = (
                facts.deployments[decision.index]
                if decision.kind == "ReadInfo"
                else (facts.cost_deployments or ())[decision.index]
            )
            if decision.kind == "ReadInfo":
                info: Final = router.registry().deployment_model_info(
                    model_id, _OPTIONAL_STRING.validate_python(row.litellm_params.get("model")) or ""
                )
                updated: Final = replace(
                    current,
                    info_loaded=True,
                    info=price_entry(_MAPPING.validate_python(info)) if info is not None else None,
                )
                return _read(
                    replace(
                        facts,
                        deployments=tuple(
                            updated if index == decision.index else item for index, item in enumerate(facts.deployments)
                        ),
                    ),
                    router,
                    rows,
                )
            keys: Final = tuple(
                _MAPPING.validate_python(_CostMap.model_validate(vars(litellm)).model_cost.get(model_id, {}))
            )
            cost: Final = replace(current, cost_keys=keys)
            return _read(
                replace(
                    facts,
                    cost_deployments=tuple(
                        cost if index == decision.index else item
                        for index, item in enumerate(facts.cost_deployments or ())
                    ),
                ),
                router,
                rows,
            )


def model_has_no_cost_mapping(model: str | None, llm_router: RegistryReader | None) -> bool:
    return unpriced_model(read_unpriced_model_facts(model, llm_router)).unpriced


def unpriced_models_in_request(model: str | Sequence[str] | None, llm_router: RegistryReader | None) -> tuple[str, ...]:
    names: Final = (model,) if isinstance(model, str) else tuple(model or ())
    return tuple(name for name in names if model_has_no_cost_mapping(name, llm_router))
