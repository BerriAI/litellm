import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.router_utils.common_utils import provider_for_generic_call
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter, PatternUtils

if TYPE_CHECKING:
    from litellm import Router

QuotaLimit: TypeAlias = Literal["rpm", "tpm"]
ProvidersOf: TypeAlias = Callable[[str], frozenset[str]]

_LIMITS: Final[tuple[QuotaLimit, ...]] = ("rpm", "tpm")
_ROUTER_PATTERNS: Final = PatternMatchRouter()


class ModelQuotaPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default: frozenset[QuotaLimit] = frozenset(_LIMITS)
    providers: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)
    models: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)


REQUIRE_BOTH_LIMITS: Final = ModelQuotaPolicy()


def parse_model_quota_setting(setting: object) -> ModelQuotaPolicy | None:
    """Any falsy value is off and any other non-mapping value requires both limits, as the boolean
    setting always behaved, so only a mapping is read as a per-provider and per-model policy."""
    if not setting:
        return None
    if is_str_keyed_mapping(setting):
        return ModelQuotaPolicy.model_validate(setting)
    return REQUIRE_BOTH_LIMITS


def _wildcard_matches(pattern: str, model: str) -> bool:
    return re.fullmatch(_ROUTER_PATTERNS.pattern_to_regex(pattern), model) is not None


def _router_specificity(pattern: str) -> tuple[int, int]:
    return PatternUtils.calculate_pattern_specificity(_ROUTER_PATTERNS.pattern_to_regex(pattern))


def _most_specific_model_pattern(model: str, entries: Mapping[str, frozenset[QuotaLimit]]) -> str | None:
    matching: Final = tuple(pattern for pattern in entries if "*" in pattern and _wildcard_matches(pattern, model))
    return max(matching, key=_router_specificity, default=None)


def required_limits(policy: ModelQuotaPolicy, model: str, providers: frozenset[str]) -> frozenset[QuotaLimit]:
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
    return frozenset[QuotaLimit]().union(*(policy.providers.get(provider, policy.default) for provider in providers))


def _is_positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def unmet_limits(required: frozenset[QuotaLimit], values: Mapping[QuotaLimit, object]) -> tuple[QuotaLimit, ...]:
    return tuple(
        limit
        for limit in _LIMITS
        if (limit in required or values.get(limit) is not None) and not _is_positive(values.get(limit))
    )


def no_providers(_model: str) -> frozenset[str]:
    return frozenset()


def deployment_provider(litellm_params: object) -> str | None:
    return provider_for_generic_call(litellm_params) if is_str_keyed_mapping(litellm_params) else None


def _routed_deployments(llm_router: "Router", model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
    named: Final[tuple[Mapping[str, object], ...]] = (
        *llm_router.get_all_deployments(model_name=model, team_id=team_id),
        *llm_router.get_model_list_from_model_alias(model_name=model),
        *llm_router.get_model_list_from_routing_groups(model_name=model),
    )
    return named or llm_router.wildcard_route_deployments(model, team_id)


def router_deployment_providers(llm_router: "Router | None", team_id: str | None) -> ProvidersOf:
    if llm_router is None:
        return no_providers

    def providers_of(model: str) -> frozenset[str]:
        return frozenset(
            provider
            for deployment in _routed_deployments(llm_router, model, team_id)
            if (provider := deployment_provider(deployment.get("litellm_params"))) is not None
        )

    return providers_of
