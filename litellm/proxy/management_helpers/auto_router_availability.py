from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, Json, TypeAdapter, ValidationError

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.router_utils.auto_router_model_naming import (
    GATED_AUTO_ROUTER_CAPABILITIES,
    capability_limit_violation,
    classify_strategy_router_model,
    count_capability_routers,
    gated_capability_of,
)
from litellm.router_utils.auto_router_tuning_baseline import (
    is_mutable_tuned_candidate,
    mutable_tuned_identities,
    tuning_quota_violation,
)
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterAllowance,
    AutoRouterAvailabilityResponse,
)


class _CatalogModelInfo(BaseModel):
    team_id: str | None = None


class _CatalogSource(BaseModel):
    model_id: str
    created_by: str | None = None
    litellm_params: Json[dict[str, object]] | dict[str, object]
    model_info: Json[_CatalogModelInfo] | _CatalogModelInfo | None = None


@dataclass(frozen=True, slots=True)
class AutoRouterCatalogEntry:
    model_id: str
    team_id: str | None
    created_by: str | None
    deployment: Mapping[str, object]


def _catalog_field(value: object, key: str) -> object:
    if not isinstance(value, str):
        return deepcopy(value)
    return decrypt_value_helper(value, key=key, exception_type="debug", return_original_value=True)


def build_auto_router_catalog(rows: Sequence[object]) -> tuple[AutoRouterCatalogEntry, ...] | None:
    try:
        sources: Final = TypeAdapter(tuple[_CatalogSource, ...]).validate_python(rows, from_attributes=True)
    except ValidationError:
        return None
    return tuple(
        AutoRouterCatalogEntry(
            model_id=row.model_id,
            team_id=row.model_info.team_id if row.model_info is not None else None,
            created_by=row.created_by,
            deployment=MappingProxyType(
                {
                    "litellm_params": MappingProxyType(
                        {
                            "model": model,
                            "complexity_router_config": _catalog_field(
                                row.litellm_params.get("complexity_router_config"), "complexity_router_config"
                            ),
                        }
                    ),
                    "model_info": MappingProxyType({"id": row.model_id, "db_model": True}),
                }
            ),
        )
        for row in sources
        if isinstance(model := _catalog_field(row.litellm_params.get("model"), "model"), str)
        and classify_strategy_router_model(model) == "complexity"
    )


def auto_router_availability(
    *,
    others: Sequence[Mapping[str, object]],
    existing: Mapping[str, object] | None,
    candidate: Mapping[str, object],
    baselines: Mapping[str, str] | None,
    limit: int | None,
) -> AutoRouterAvailabilityResponse:
    existing_params: Final = None if existing is None else existing.get("litellm_params")
    candidate_params: Final = candidate.get("litellm_params")
    owned: Final = gated_capability_of(existing_params) if isinstance(existing_params, Mapping) else None
    claimed: Final = gated_capability_of(candidate_params) if isinstance(candidate_params, Mapping) else None
    counts: Final = tuple(
        (capability, count_capability_routers(others, capability=capability))
        for capability in GATED_AUTO_ROUTER_CAPABILITIES
    )
    tuned_count: Final = len(mutable_tuned_identities(others, baselines)) if baselines is not None else 0
    allowances: Final = tuple(
        AutoRouterAllowance(
            key=capability.key,
            limit=limit,
            remaining=None if limit is None else max(0, limit - held),
            used_by_this_router=owned is capability,
        )
        for capability, held in counts
    )
    capability_error: Final = next(
        (
            capability_limit_violation(capability=capability, held=held + 1, limit=limit)
            for capability, held in counts
            if capability is claimed
        ),
        None,
    )
    tuning_error: Final = (
        tuning_quota_violation(candidate=candidate, others=others, baselines=baselines, limit=limit)
        if baselines is not None
        else None
    )
    capability_labels: Final = {
        "heuristic_v2": "Heuristic v2",
        "capability": "Capability",
        "llm_v2": "Fuse v2",
        "tier_or_classifier_prompt": "Custom tiers or classifier instructions",
    }
    return AutoRouterAvailabilityResponse(
        allowances=(
            *allowances,
            AutoRouterAllowance(
                key="heuristic_tuning",
                limit=limit,
                remaining=None if limit is None or baselines is None else max(0, limit - tuned_count),
                available=limit is None or baselines is not None,
                used_by_this_router=bool(
                    existing is not None and baselines is not None and is_mutable_tuned_candidate(existing, baselines)
                ),
            ),
        ),
        error=(
            f"{capability_labels[claimed.key]} has no available allowance. Choose another option or free an existing allowance."
            if capability_error is not None and claimed is not None
            else "These scoring rules need an available Rule-based tuning allowance. Check the weights, thresholds, keywords, and custom dimensions in Advanced settings. Model choices do not use this allowance."
            if tuning_error is not None
            else None
        ),
    )
