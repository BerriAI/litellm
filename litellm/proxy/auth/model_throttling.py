import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final

from pydantic import ConfigDict, Field

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    model_access_group_spend_counter_key,
    project_spend_counter_key,
)
from litellm.router import Router
from litellm.router_utils.common_utils import resolve_served_model
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.model_throttling import ActiveModelThrottle, ModelThrottling
from litellm.types.utils import CallTypes


class _MemberDefaults(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    team_member_budget_id: str | None = None


class _TeamDefaults(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    metadata: _MemberDefaults | None = None


class _RouterSettings(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    model_group_alias: Mapping[str, object] | None = None


class _Aliases(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    aliases: Mapping[str, str] = Field(default_factory=dict)
    router_settings: _RouterSettings | None = None


@dataclass(frozen=True, slots=True)
class ModelThrottleSource:
    scope: str
    counter_key: str
    spend: float
    max_budget: float | None
    config: ModelThrottling | None
    fallback_authoritative: bool = False


def budget_source(
    scope: str, counter_key: str, spend: float, budget: LiteLLM_BudgetTable | None
) -> ModelThrottleSource:
    return ModelThrottleSource(
        scope,
        counter_key,
        spend,
        budget.max_budget if budget is not None else None,
        budget.model_throttling if budget is not None else None,
        scope in ("tag", "end user"),
    )


async def active_throttles(source: ModelThrottleSource) -> tuple[ActiveModelThrottle, ...]:
    if source.config is None or source.max_budget is None:
        return ()
    from litellm.proxy.proxy_server import get_current_spend

    spend: Final = await get_current_spend(
        counter_key=source.counter_key,
        fallback_spend=source.spend,
        max_budget=source.max_budget,
        fallback_authoritative=source.fallback_authoritative,
    )
    return throttles_at_spend(source, spend)


def throttles_at_spend(source: ModelThrottleSource, spend: float) -> tuple[ActiveModelThrottle, ...]:
    if source.config is None:
        return ()
    return tuple(
        ActiveModelThrottle(scope=source.scope, spend=spend, threshold=threshold)
        for threshold in source.config.thresholds
        if spend >= threshold.spend
    )


async def inherited_sources(
    valid_token: UserAPIKeyAuth,
    team: LiteLLM_TeamTable | None,
    membership: LiteLLM_TeamMembership | None,
) -> tuple[ModelThrottleSource, ...]:
    from litellm.proxy.auth.auth_checks import get_org_object, get_team_member_default_budget
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    org_id: Final = valid_token.org_id or (team.organization_id if team is not None else None)
    org: Final = (
        await get_org_object(
            org_id=org_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            include_budget_table=True,
        )
        if org_id is not None and prisma_client is not None
        else None
    )
    member_budget: Final = membership.litellm_budget_table if membership is not None else None
    defaults: Final = _TeamDefaults.model_validate(team).metadata if team is not None else None
    default_id: Final = defaults.team_member_budget_id if defaults is not None else None
    default_budget: Final = (
        await get_team_member_default_budget(
            budget_id=default_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            raise_on_lookup_error=True,
        )
        if isinstance(default_id, str)
        and valid_token.user_id is not None
        and (member_budget is None or member_budget.max_budget is None)
        else None
    )
    return (
        budget_source("organization", f"spend:org:{org_id}", org.spend, org.litellm_budget_table)
        if org is not None
        else budget_source("organization", "", 0, None),
        budget_source(
            "team member",
            f"spend:team_member:{valid_token.user_id}:{valid_token.team_id}",
            (membership.spend or 0) if membership is not None else 0,
            default_budget or member_budget,
        ),
    )


async def tagged_sources(token: UserAPIKeyAuth, tags: Sequence[str]) -> tuple[ModelThrottleSource, ...]:
    from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch, get_tag_objects_batch
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        return ()
    tag_objects: Final = (
        await get_tag_objects_batch(
            tag_names=tags,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if tags
        else {}
    )
    groups: Final = (
        await get_model_access_group_budgets_batch(
            access_group_names=token.matched_model_access_groups,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
        if token.matched_model_access_groups
        else {}
    )
    return (
        *(
            budget_source("tag", f"spend:tag:{name}", tag.spend or 0, tag.litellm_budget_table)
            for name, tag in tag_objects.items()
        ),
        *(
            ModelThrottleSource(
                "model access group",
                model_access_group_spend_counter_key(name),
                group.spend,
                group.max_budget,
                group.model_throttling,
                True,
            )
            for name, group in groups.items()
        ),
    )


async def load_model_throttles(
    *,
    valid_token: UserAPIKeyAuth,
    team: LiteLLM_TeamTable | None,
    user: LiteLLM_UserTable | None,
    end_user: LiteLLM_EndUserTable | None,
    project: LiteLLM_ProjectTableCachedObj | None,
    membership: LiteLLM_TeamMembership | None,
    apply_user_budget_to_team_keys: bool,
    tags: Sequence[str] = (),
) -> tuple[ActiveModelThrottle, ...]:
    sources: Final = (
        ModelThrottleSource(
            "key",
            f"spend:key:{valid_token.token}",
            valid_token.spend or 0,
            valid_token.max_budget,
            valid_token.model_throttling,
        ),
        ModelThrottleSource(
            "team", f"spend:team:{team.team_id}", team.spend or 0, team.max_budget, team.model_throttling
        )
        if team is not None
        else None,
        ModelThrottleSource(
            "user", f"spend:user:{user.user_id}", user.spend or 0, user.max_budget, user.model_throttling
        )
        if user is not None and (team is None or apply_user_budget_to_team_keys)
        else None,
        budget_source(
            "project", project_spend_counter_key(project.project_id), project.spend or 0, project.litellm_budget_table
        )
        if project is not None
        else None,
        budget_source(
            "end user", f"spend:end_user:{end_user.user_id}", end_user.spend or 0, end_user.litellm_budget_table
        )
        if end_user is not None
        else None,
        *await inherited_sources(valid_token, team, membership),
        *await tagged_sources(valid_token, tags),
    )
    groups: Final = await asyncio.gather(*(active_throttles(source) for source in sources if source is not None))
    return tuple(chain.from_iterable(groups))


def canonical_model(model: str, token: UserAPIKeyAuth, router: Router | None) -> str:
    aliases: Final = _Aliases.model_validate(token)
    settings: Final = aliases.router_settings
    served: Final = resolve_served_model(
        model,
        litellm.model_alias_map,
        aliases.aliases,
        settings.model_group_alias
        if settings is not None and settings.model_group_alias is not None
        else router.model_group_alias
        if router is not None
        else None,
    )
    team_alias: Final[object] = (token.team_model_aliases or {}).get(served)
    resolved: Final = team_alias if isinstance(team_alias, str) else served
    deployment: Final = router.get_deployment(resolved) if router is not None else None
    if deployment is not None:
        return deployment.model_name
    return router.routable_model_group(resolved) if router is not None else resolved


def model_throttle_for(
    model: str,
    token: UserAPIKeyAuth,
    router: Router | None,
) -> ActiveModelThrottle | None:
    if not token.active_model_throttles:
        return None
    requested: Final = canonical_model(model, token, router)
    for throttle in token.active_model_throttles:
        if any(canonical_model(paused, token, router) == requested for paused in throttle.threshold.models):
            return throttle
    return None


def enforce_model_throttling(
    models: Sequence[str],
    token: UserAPIKeyAuth,
    router: Router | None,
) -> None:
    for model in models:
        throttle = model_throttle_for(model, token, router)
        if throttle is not None:
            raise ProxyException(
                message=(
                    f"Model '{model}' is paused by {throttle.scope} model throttling after "
                    f"${throttle.threshold.spend:g} of spend. Choose another available model "
                    "or wait for the budget to reset."
                ),
                type="model_throttled",
                param="model",
                code=429,
            )


class _DeploymentInfo(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    id: str | None = None


class _RequestMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    user_api_key_auth: UserAPIKeyAuth | None = None


class _DeploymentRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    model_info: _DeploymentInfo | None = None
    metadata: _RequestMetadata | None = None
    litellm_metadata: _RequestMetadata | None = None


class ModelThrottlingHook(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        from litellm.proxy.proxy_server import llm_router

        request: Final = _DeploymentRequest.model_validate(kwargs)
        metadata: Final = request.litellm_metadata or request.metadata
        token: Final = metadata.user_api_key_auth if metadata is not None else None
        if token is None or not token.active_model_throttles or request.model_info is None:
            return
        if request.model_info.id is not None:
            enforce_model_throttling((request.model_info.id,), token, llm_router)
