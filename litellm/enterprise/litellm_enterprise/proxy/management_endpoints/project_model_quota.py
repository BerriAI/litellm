from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.proxy.management_helpers.model_quota_policy import (
    EVERY_LIMIT,
    ModelQuotaPolicy,
    QuotaLimit,
    RouteOf,
    required_limits,
    unmet_limits,
)

_NO_LIMITS_SET: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class MissingModelQuota:
    model: str
    limits: tuple[QuotaLimit, ...]


def _limits_set_for(
    model: str, rpm_set: Mapping[str, object], tpm_set: Mapping[str, object]
) -> Mapping[QuotaLimit, object]:
    return MappingProxyType({"rpm": rpm_set.get(model), "tpm": tpm_set.get(model)})


def _limits_in(limits: object) -> Mapping[str, object]:
    return limits if is_str_keyed_mapping(limits) else _NO_LIMITS_SET


def missing_project_model_quota(
    models: Sequence[str], metadata: Mapping[str, object], policy: ModelQuotaPolicy, route_of: RouteOf
) -> tuple[MissingModelQuota, ...]:
    rpm_set: Final = _limits_in(metadata.get("model_rpm_limit"))
    tpm_set: Final = _limits_in(metadata.get("model_tpm_limit"))
    gaps: Final = tuple(
        MissingModelQuota(
            model,
            unmet_limits(required_limits(policy, model, route_of(model)), _limits_set_for(model, rpm_set, tpm_set)),
        )
        for model in models
    )
    return tuple(gap for gap in gaps if gap.limits)


def models_short_of_a_limit(models: Sequence[str], metadata: Mapping[str, object]) -> tuple[str, ...]:
    rpm_set: Final = _limits_in(metadata.get("model_rpm_limit"))
    tpm_set: Final = _limits_in(metadata.get("model_tpm_limit"))
    return tuple(model for model in models if unmet_limits(EVERY_LIMIT, _limits_set_for(model, rpm_set, tpm_set)))


def describe_missing_quota(missing: Sequence[MissingModelQuota]) -> str:
    return ", ".join(f"{gap.model} ({' and '.join(f'model_{limit}_limit' for limit in gap.limits)})" for gap in missing)
