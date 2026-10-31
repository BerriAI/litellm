import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

import litellm
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider, is_openai_compatible_endpoint
from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter, PatternUtils
from litellm.litellm_core_utils.routing_common_utils import provider_for_generic_call
from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm import Router

QuotaLimit: TypeAlias = Literal["rpm", "tpm"]

ENFORCE_PROJECT_MODEL_QUOTA_SETTING: Final = "enforce_project_model_quota"

_LIMITS: Final[tuple[QuotaLimit, ...]] = ("rpm", "tpm")
_ROUTER_PATTERNS: Final = PatternMatchRouter()


def _is_known_provider(provider: str) -> bool:
    return provider in litellm.provider_list or JSONProviderRegistry.exists(provider)


class ModelQuotaPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default: frozenset[QuotaLimit] = frozenset(_LIMITS)
    providers: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)
    models: Mapping[str, frozenset[QuotaLimit]] = Field(default_factory=dict)

    @field_validator("providers")
    @classmethod
    def _known_providers(cls, providers: Mapping[str, frozenset[QuotaLimit]]) -> Mapping[str, frozenset[QuotaLimit]]:
        unknown: Final = sorted(provider for provider in providers if not _is_known_provider(provider))
        if unknown:
            raise ValueError(f"unknown provider ids {unknown}")
        return providers


REQUIRE_BOTH_LIMITS: Final = ModelQuotaPolicy()
_POLICY_KEYS: Final = frozenset(ModelQuotaPolicy.model_fields)


@dataclass(frozen=True, slots=True)
class TeamDeployment:
    provider: str | None
    called_model: str | None


@dataclass(frozen=True, slots=True)
class ModelRoute:
    providers: frozenset[str] = frozenset()
    team_deployments: frozenset[TeamDeployment] = frozenset()


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


def _provider_limits(policy: ModelQuotaPolicy, provider: str | None) -> frozenset[QuotaLimit]:
    return policy.default if provider is None else policy.providers.get(provider, policy.default)


def _team_deployment_limits(policy: ModelQuotaPolicy, deployment: TeamDeployment) -> frozenset[QuotaLimit]:
    entry: Final = None if deployment.called_model is None else _model_entry(policy.models, deployment.called_model)
    return _provider_limits(policy, deployment.provider) if entry is None else entry


def required_limits(policy: ModelQuotaPolicy, model: str, route: ModelRoute) -> frozenset[QuotaLimit]:
    """A model entry (exact name, then the most specific `*` pattern) wins over the providers of the
    model's deployments, which win over the default. A model served by several providers needs every
    limit any of them requires. A team picks its model's public name, so a team deployment matches
    model entries by the model it calls instead."""
    entry: Final = None if route.team_deployments else _model_entry(policy.models, model)
    if entry is not None:
        return entry
    if not route.providers and not route.team_deployments:
        return policy.default
    return frozenset[QuotaLimit]().union(
        *(_team_deployment_limits(policy, deployment) for deployment in route.team_deployments),
        *(_provider_limits(policy, provider) for provider in route.providers),
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


def _resolved_api_base(api_base: object) -> object:
    if isinstance(api_base, str) and api_base.startswith("os.environ/"):
        return get_secret_str(api_base)
    return api_base


def _provider_behind_api_base(litellm_params: Mapping[str, object]) -> str | None:
    model: Final = litellm_params.get("model")
    api_base: Final = _resolved_api_base(litellm_params.get("api_base"))
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


def _called_model(litellm_params: Mapping[str, object]) -> str | None:
    model: Final = litellm_params.get("model")
    if not isinstance(model, str):
        return None
    prefix, _, rest = model.partition("/")
    return rest if rest and _is_known_provider(prefix) else model


def team_deployment(litellm_params: object) -> TeamDeployment:
    if not is_str_keyed_mapping(litellm_params):
        return TeamDeployment(provider=None, called_model=None)
    return TeamDeployment(provider=deployment_provider(litellm_params), called_model=_called_model(litellm_params))


def deployment_route(litellm_params: Mapping[str, object], team_model: bool) -> ModelRoute:
    if team_model:
        return ModelRoute(team_deployments=frozenset((team_deployment(litellm_params),)))
    provider: Final = deployment_provider(litellm_params)
    return ModelRoute(providers=frozenset() if provider is None else frozenset((provider,)))


def _named_deployments(llm_router: "Router", model: str, team_id: str | None) -> tuple[Mapping[str, object], ...]:
    return (
        *llm_router.registry().deployments(model, source="all", team_id=team_id),
        *llm_router.registry().deployments(model, source="routing_groups"),
    )


def _routed_deployments(llm_router: "Router", model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
    target: Final = llm_router.registry().resolve(model)
    return _named_deployments(llm_router, target, team_id) or llm_router.registry().deployments(
        target, source="wildcard", team_id=team_id
    )


def _is_team_deployment(deployment: Mapping[str, object]) -> bool:
    model_info: Final = deployment.get("model_info")
    return is_str_keyed_mapping(model_info) and model_info.get("team_id") is not None


def router_model_routes(llm_router: "Router | None", team_id: str | None) -> RouteOf:
    if llm_router is None:
        return no_route

    def route_of(model: str) -> ModelRoute:
        deployments: Final = _routed_deployments(llm_router, model, team_id)
        shared: Final = tuple(deployment for deployment in deployments if not _is_team_deployment(deployment))
        return ModelRoute(
            providers=frozenset(
                provider
                for deployment in shared
                if (provider := deployment_provider(deployment.get("litellm_params"))) is not None
            ),
            team_deployments=frozenset(
                team_deployment(deployment.get("litellm_params"))
                for deployment in deployments
                if _is_team_deployment(deployment)
            ),
        )

    return route_of


def _deployment_id(deployment: Mapping[str, object]) -> object:
    model_info: Final = deployment.get("model_info")
    return model_info.get("id") if is_str_keyed_mapping(model_info) else None


def _routable_name(llm_router: "Router | None", model: str) -> str:
    return model if llm_router is None else llm_router.registry().resolve(model)


def _patterns_tried_before(patterns: Sequence[str], regex: str) -> Sequence[str]:
    if regex in patterns:
        return patterns[: patterns.index(regex)]
    specificity: Final = PatternUtils.calculate_pattern_specificity(regex)
    return tuple(rival for rival in patterns if PatternUtils.calculate_pattern_specificity(rival) >= specificity)


def _team_pattern_serves(llm_router: "Router", team_id: str, pattern: str, model: str) -> bool:
    regex: Final = _ROUTER_PATTERNS.pattern_to_regex(pattern)
    if re.match(regex, model) is None or _named_deployments(llm_router, model, team_id):
        return False
    team_router: Final = llm_router.registry().patterns(team=True).get(team_id)
    patterns: Final = () if team_router is None else tuple(team_router.patterns)
    return not any(re.match(rival, model) is not None for rival in _patterns_tried_before(patterns, regex))


def team_write_serves(llm_router: "Router | None", team_id: str, public_name: str, model: str) -> bool:
    """Whether a request from the team for ``model`` reaches the team route named ``public_name``: the name
    itself or an alias of it, or, for a `*` name, a model no named deployment shadows and no team pattern the
    router tries first takes."""
    target: Final = _routable_name(llm_router, model)
    if "*" not in public_name or llm_router is None:
        return target == public_name
    return _team_pattern_serves(llm_router, team_id, public_name, target)


def _served_params(litellm_params: object, public_name: str, model: str) -> object:
    if "*" not in public_name or not is_str_keyed_mapping(litellm_params):
        return litellm_params
    matched: Final = re.match(_ROUTER_PATTERNS.pattern_to_regex(public_name), model)
    called: Final = litellm_params.get("model")
    if matched is None or not isinstance(called, str):
        return litellm_params
    return MappingProxyType(
        {
            **litellm_params,
            "model": PatternMatchRouter.set_deployment_model_name(
                matched_pattern=matched, litellm_deployment_litellm_model=called
            ),
        }
    )


def team_route_after_write(
    llm_router: "Router | None",
    team_id: str,
    public_name: str,
    replaced_id: str | None,
    written_params: Mapping[str, object] | None,
    model: str,
) -> ModelRoute | None:
    deployments: Final = (
        () if llm_router is None else llm_router.registry().deployments(public_name, source="all", team_id=team_id)
    )
    kept: Final = tuple(
        deployment.get("litellm_params")
        for deployment in deployments
        if _is_team_deployment(deployment) and _deployment_id(deployment) != replaced_id
    )
    team_params: Final = kept if written_params is None else (*kept, written_params)
    target: Final = _routable_name(llm_router, model)
    if team_params:
        return ModelRoute(
            team_deployments=frozenset(
                team_deployment(_served_params(litellm_params, public_name, target)) for litellm_params in team_params
            )
        )
    if llm_router is None or not _routed_deployments(llm_router, model, None):
        return None
    return router_model_routes(llm_router, None)(model)


def limits_added_by_team_write(
    policy: ModelQuotaPolicy,
    llm_router: "Router | None",
    team_id: str,
    public_name: str,
    replaced_id: str | None,
    written_params: Mapping[str, object] | None,
    model: str,
) -> frozenset[QuotaLimit]:
    after: Final = team_route_after_write(llm_router, team_id, public_name, replaced_id, written_params, model)
    if after is None:
        return frozenset()
    now: Final = router_model_routes(llm_router, team_id)(model)
    return required_limits(policy, model, after) - required_limits(policy, model, now)


def exempts_any_route(policy: ModelQuotaPolicy) -> bool:
    return any(
        limits != REQUIRE_BOTH_LIMITS.default
        for limits in (policy.default, *policy.providers.values(), *policy.models.values())
    )
