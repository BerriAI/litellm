from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.proxy._types import ModelAccessDeniedProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _check_agent_access_group_model_access,  # pyright: ignore[reportPrivateUsage]  # Reuse native model policy checks without duplicating them
    _check_agent_caller_model_access,  # pyright: ignore[reportPrivateUsage]  # Reuse native model policy checks without duplicating them
    _user_object_or_none,  # pyright: ignore[reportPrivateUsage]  # Reuse native model policy checks without duplicating them
    can_key_call_resolved_model,
    can_user_call_model,
    get_project_object,
    get_team_object,
    key_model_aliases_for_auth_check,
)
from litellm.proxy.auth.team_grants import team_model_aliases
from litellm.proxy.common_utils.model_listing_utils import TeamModelNameTranslator
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router import Router
from litellm.types.router import ModelInfo


async def effective_key_models(
    key: UserAPIKeyAuth,
    router: Router | None,
    prisma_client: PrismaClient,
    cache: UserApiKeyCache,
    logging: ProxyLogging,
    settings: Mapping[str, object] = MappingProxyType({}),
) -> tuple[str, ...]:
    if router is None:
        return ()
    team: Final = (
        await get_team_object(
            team_id=key.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            proxy_logging_obj=logging,
        )
        if key.team_id
        else None
    )
    project: Final = (
        await get_project_object(
            project_id=key.project_id,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            proxy_logging_obj=logging,
        )
        if key.project_id
        else None
    )
    if (team is not None and team.blocked) or (project is not None and project.blocked):
        return ()
    target: Final = UserAPIKeyAuth.model_validate(
        MappingProxyType(
            {
                **key.model_dump(),
                "team_models": team.models if team else (),
                "team_model_aliases": team_model_aliases(team),
                "user_role": None,
            }
        )
    )
    owner: Final = (
        await _user_object_or_none(target, prisma_client, cache, logging)
        if target.team_id is None and target.user_id is not None
        else None
    )
    registrations: Final = router.get_model_list() or ()
    candidates: Final = frozenset(row["model_name"] for row in registrations)

    async def allowed(model: str) -> bool:
        if "*" in model:
            return False
        deployments: Final = router.get_model_list(model_name=model, team_id=target.team_id) or ()
        if not any(
            ModelInfo.model_validate(deployment.get("model_info") or MappingProxyType({})).team_id  # pyright: ignore[reportUnknownMemberType]  # Validate the router’s legacy untyped model_info here
            in (None, target.team_id)
            for deployment in deployments
        ):
            return False
        try:
            await can_key_call_resolved_model(model, None, target, router)
            await _check_agent_access_group_model_access(model=model, valid_token=target, llm_router=router)
            await _check_agent_caller_model_access(
                model=model,
                valid_token=target,
                llm_router=router,
                prisma_client=prisma_client,
                user_api_key_cache=cache,
                proxy_logging_obj=logging,
            )
            await can_user_call_model(
                model=model,
                llm_router=router,
                user_object=owner,
                key_model_aliases=key_model_aliases_for_auth_check(target),
            )
        except ModelAccessDeniedProxyException:
            return False
        return True

    eligible: Final = tuple(frozenset([model for model in candidates if await allowed(model)]))
    entries: Final = TeamModelNameTranslator.listing_entries(eligible, router, settings)
    return tuple(sorted(public_name for public_name, _ in entries))
