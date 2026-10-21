import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.router_utils.common_utils import provider_for_generic_call
from litellm.router_utils.pattern_match_deployments import PatternUtils

if TYPE_CHECKING:
    from litellm import Router

QuotaLimit: TypeAlias = Literal["rpm", "tpm"]
ProvidersOf: TypeAlias = Callable[[str], frozenset[str]]

ENFORCE_PROJECT_MODEL_QUOTA_SETTING: Final = "enforce_project_model_quota"
_BOTH_LIMITS: Final[frozenset[QuotaLimit]] = frozenset(("rpm", "tpm"))
_NO_LIMITS_SET: Final[Mapping[str, object]] = MappingProxyType({})


class ProjectModelQuotaPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default: frozenset[QuotaLimit] = _BOTH_LIMITS
    providers: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)
    models: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)


REQUIRE_BOTH_LIMITS: Final = ProjectModelQuotaPolicy()

_QUOTA_SETTING: Final = TypeAdapter(bool | ProjectModelQuotaPolicy | None)


@dataclass(frozen=True, slots=True)
class MissingModelQuota:
    model: str
    limits: tuple[QuotaLimit, ...]


def parse_project_model_quota_setting(setting: object) -> ProjectModelQuotaPolicy | None:
    parsed: Final = _QUOTA_SETTING.validate_python(setting)
    match parsed:
        case ProjectModelQuotaPolicy():
            return parsed
        case True:
            return REQUIRE_BOTH_LIMITS
        case _:
            return None


def _wildcard_matches(pattern: str, model: str) -> bool:
    return re.fullmatch(re.escape(pattern).replace(r"\*", ".*"), model) is not None


def _most_specific_model_pattern(model: str, entries: Mapping[str, frozenset[QuotaLimit]]) -> str | None:
    matching: Final = tuple(pattern for pattern in entries if "*" in pattern and _wildcard_matches(pattern, model))
    return max(matching, key=PatternUtils.calculate_pattern_specificity, default=None)


def required_limits(policy: ProjectModelQuotaPolicy, model: str, providers: frozenset[str]) -> frozenset[QuotaLimit]:
    """A model entry (exact name, then the most specific `*` pattern) wins over the providers of the
    model's deployments, which win over the default. A model served by several providers needs every
    limit any of them requires."""
    exact: Final = policy.models.get(model)
    if exact is not None:
        return exact
    pattern: Final = _most_specific_model_pattern(model, policy.models)
    if pattern is not None:
        return policy.models[pattern]
    if not providers:
        return policy.default
    return frozenset(chain.from_iterable(policy.providers.get(provider, policy.default) for provider in providers))


def _is_positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unset_limits(
    model: str, required: frozenset[QuotaLimit], set_limits: Mapping[QuotaLimit, Mapping[str, object]]
) -> tuple[QuotaLimit, ...]:
    return tuple(sorted(limit for limit in required if not _is_positive(set_limits[limit].get(model))))


def missing_project_model_quota(
    models: Sequence[str],
    rpm_limits: Mapping[str, object] | None,
    tpm_limits: Mapping[str, object] | None,
    policy: ProjectModelQuotaPolicy,
    providers_of: ProvidersOf,
) -> tuple[MissingModelQuota, ...]:
    set_limits: Final[Mapping[QuotaLimit, Mapping[str, object]]] = MappingProxyType(
        {"rpm": rpm_limits or _NO_LIMITS_SET, "tpm": tpm_limits or _NO_LIMITS_SET}
    )
    gaps: Final = tuple(
        MissingModelQuota(model, _unset_limits(model, required_limits(policy, model, providers_of(model)), set_limits))
        for model in models
    )
    return tuple(gap for gap in gaps if gap.limits)


def describe_missing_quota(missing: Sequence[MissingModelQuota]) -> str:
    return ", ".join(f"{gap.model} ({' and '.join(f'model_{limit}_limit' for limit in gap.limits)})" for gap in missing)


def no_providers(_model: str) -> frozenset[str]:
    return frozenset()


def router_deployment_providers(llm_router: "Router | None", team_id: str | None) -> ProvidersOf:
    if llm_router is None:
        return no_providers

    def providers_of(model: str) -> frozenset[str]:
        deployments: Final = llm_router.get_model_list(model_name=model, team_id=team_id) or ()
        return frozenset(
            provider
            for deployment in deployments
            if (provider := provider_for_generic_call(deployment["litellm_params"])) is not None
        )

    return providers_of
