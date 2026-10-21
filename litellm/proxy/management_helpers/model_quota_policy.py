import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider, is_openai_compatible_endpoint
from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.router_utils.common_utils import provider_for_generic_call
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter, PatternUtils

if TYPE_CHECKING:
    from litellm import Router

QuotaLimit: TypeAlias = Literal["rpm", "tpm"]

_LIMITS: Final[tuple[QuotaLimit, ...]] = ("rpm", "tpm")
_ROUTER_PATTERNS: Final = PatternMatchRouter()


class ModelQuotaPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default: frozenset[QuotaLimit] = frozenset(_LIMITS)
    providers: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)
    models: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)


REQUIRE_BOTH_LIMITS: Final = ModelQuotaPolicy()
_POLICY_KEYS: Final = frozenset(ModelQuotaPolicy.model_fields)


@dataclass(frozen=True, slots=True)
class ModelRoute:
    providers: frozenset[str] = frozenset()
    team_named: bool = False


RouteOf: TypeAlias = Callable[[str], ModelRoute]
NO_ROUTE: Final = ModelRoute()


def parse_model_quota_setting(setting: object) -> ModelQuotaPolicy | None:
    """Any falsy value is off and any other value requires both limits, as the boolean setting always
    behaved, so only a mapping that names a policy key is read as a per-provider and per-model policy."""
    if not setting:
        return None
    if is_str_keyed_mapping(setting) and not _POLICY_KEYS.isdisjoint(setting):
        return ModelQuotaPolicy.model_validate(setting)
    return REQUIRE_BOTH_LIMITS


def _wildcard_matches(pattern: str, model: str) -> bool:
    return re.match(_ROUTER_PATTERNS.pattern_to_regex(pattern), model) is not None


def _router_specificity(pattern: str) -> tuple[int, int]:
    return PatternUtils.calculate_pattern_specificity(_ROUTER_PATTERNS.pattern_to_regex(pattern))


def _most_specific_model_pattern(model: str, entries: Mapping[str, frozenset[QuotaLimit]]) -> str | None:
    matching: Final = tuple(pattern for pattern in entries if "*" in pattern and _wildcard_matches(pattern, model))
    return max(matching, key=_router_specificity, default=None)


def _model_entry(entries: Mapping[str, frozenset[QuotaLimit]], model: str) -> frozenset[QuotaLimit] | None:
    exact: Final = entries.get(model)
    if exact is not None:
        return exact
    pattern: Final = _most_specific_model_pattern(model, entries)
    return None if pattern is None else entries[pattern]


def required_limits(policy: ModelQuotaPolicy, model: str, route: ModelRoute) -> frozenset[QuotaLimit]:
    """A model entry (exact name, then the most specific `*` pattern) wins over the providers of the
    model's deployments, which win over the default. A model served by several providers needs every
    limit any of them requires. A team model's name is picked by its team, so model entries never match it."""
    entry: Final = None if route.team_named else _model_entry(policy.models, model)
    if entry is not None:
        return entry
    if not route.providers:
        return policy.default
    return frozenset[QuotaLimit]().union(
        *(policy.providers.get(provider, policy.default) for provider in route.providers)
    )


def _is_positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def unmet_limits(required: frozenset[QuotaLimit], values: Mapping[QuotaLimit, object]) -> tuple[QuotaLimit, ...]:
    return tuple(
        limit
        for limit in _LIMITS
        if (limit in required or values.get(limit) is not None) and not _is_positive(values.get(limit))
    )


def no_route(_model: str) -> ModelRoute:
    return NO_ROUTE


def _provider_behind_api_base(litellm_params: Mapping[str, object]) -> str | None:
    model: Final = litellm_params.get("model")
    api_base: Final = litellm_params.get("api_base")
    if litellm_params.get("custom_llm_provider") or not isinstance(model, str) or not isinstance(api_base, str):
        return None
    if model.split("/", 1)[0] in litellm.provider_list or not is_openai_compatible_endpoint(api_base):
        return None
    try:
        _, provider, _, _ = get_llm_provider(model=model, api_base=api_base)
    except BadRequestError:
        return None
    return provider


def deployment_provider(litellm_params: object) -> str | None:
    if not is_str_keyed_mapping(litellm_params):
        return None
    return _provider_behind_api_base(litellm_params) or provider_for_generic_call(litellm_params)


def _routed_deployments(llm_router: "Router", model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
    named: Final[tuple[Mapping[str, object], ...]] = (
        *llm_router.get_all_deployments(model_name=model, team_id=team_id),
        *llm_router.get_model_list_from_model_alias(model_name=model),
        *llm_router.get_model_list_from_routing_groups(model_name=model),
    )
    return named or llm_router.wildcard_route_deployments(model, team_id)


def _is_team_deployment(deployment: Mapping[str, object]) -> bool:
    model_info: Final = deployment.get("model_info")
    return is_str_keyed_mapping(model_info) and model_info.get("team_id") is not None


def router_model_routes(llm_router: "Router | None", team_id: str | None) -> RouteOf:
    if llm_router is None:
        return no_route

    def route_of(model: str) -> ModelRoute:
        deployments: Final = _routed_deployments(llm_router, model, team_id)
        return ModelRoute(
            providers=frozenset(
                provider
                for deployment in deployments
                if (provider := deployment_provider(deployment.get("litellm_params"))) is not None
            ),
            team_named=any(_is_team_deployment(deployment) for deployment in deployments),
        )

    return route_of
