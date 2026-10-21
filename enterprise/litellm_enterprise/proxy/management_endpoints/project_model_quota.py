from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.proxy.management_helpers.model_quota_policy import (
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


def missing_project_model_quota(
    models: Sequence[str],
    rpm_limits: Mapping[str, object] | None,
    tpm_limits: Mapping[str, object] | None,
    policy: ModelQuotaPolicy,
    route_of: RouteOf,
) -> tuple[MissingModelQuota, ...]:
    rpm_set: Final = rpm_limits or _NO_LIMITS_SET
    tpm_set: Final = tpm_limits or _NO_LIMITS_SET
    gaps: Final = tuple(
        MissingModelQuota(
            model,
            unmet_limits(required_limits(policy, model, route_of(model)), _limits_set_for(model, rpm_set, tpm_set)),
        )
        for model in models
    )
    return tuple(gap for gap in gaps if gap.limits)


def describe_missing_quota(missing: Sequence[MissingModelQuota]) -> str:
    return ", ".join(f"{gap.model} ({' and '.join(f'model_{limit}_limit' for limit in gap.limits)})" for gap in missing)
