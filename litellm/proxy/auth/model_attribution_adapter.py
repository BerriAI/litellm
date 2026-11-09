from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Final

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.model_access import key_model_grants
from litellm._auth.model_attribution import ModelGroupAttributionFacts, model_group_attribution
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_ProjectTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Sources:
    model: tuple[str, ...]
    token: UserAPIKeyAuth
    team: LiteLLM_TeamTable | None
    project: LiteLLM_ProjectTable | None
    router: RegistryReader
    prisma: PrismaClient
    cache: UserApiKeyCache
    logging: ProxyLogging
    membership: LiteLLM_TeamMembership | None


async def read_model_group_attribution_facts(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    project_object: LiteLLM_ProjectTable | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
) -> ModelGroupAttributionFacts:
    facts: Final = ModelGroupAttributionFacts(
        model_present=model is not None,
        token_present=valid_token is not None,
        router_present=llm_router is not None,
        database_present=prisma_client is not None,
        team_present=team_object is not None,
        user_present=valid_token is not None and valid_token.user_id is not None,
        membership_loaded=team_membership_loaded,
        token_org=valid_token.org_id if valid_token is not None else None,
        team_org=team_object.organization_id if team_object is not None else None,
    )
    if model_group_attribution(facts).kind == "Complete":
        return facts
    if model is None or valid_token is None or llm_router is None or prisma_client is None:
        raise NativeAuthRuntimeError("attribution read requested without its source")
    return await _read(
        facts,
        _Sources(
            (model,) if isinstance(model, str) else tuple(model),
            valid_token,
            team_object,
            project_object,
            llm_router,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
            team_membership,
        ),
    )


def _covering_groups(source: _Sources) -> Iterator[str]:
    for model in source.model:
        yield from source.router.registry().access_groups(model_name=model, team_id=source.token.team_id)


async def _read(facts: ModelGroupAttributionFacts, source: _Sources) -> ModelGroupAttributionFacts:
    from litellm.proxy.auth.auth_checks import (
        _load_model_access_group_registry,  # pyright: ignore[reportPrivateUsage]  # reuse the existing registry IO without its attribution policy
    )

    decision: Final = model_group_attribution(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "ReadRegistry":
            registry: Final = await _load_model_access_group_registry(source.prisma, source.cache)
            return await _read(
                replace(facts, registry_loaded=True, registry=tuple(registry) if registry is not None else None), source
            )
        case "ReadCovering":
            return await _read(replace(facts, covering=tuple(_covering_groups(source))), source)
        case "ReadKey":
            key: Final = key_model_grants(
                object_model_grants(source.token), source.token.team_id, tuple(source.token.team_models or ())
            )
            return await _read(replace(facts, grants=(*facts.grants, key)), source)
        case "ReadTeam":
            team: Final = object_model_grants(source.team) if source.team is not None else ()
            return await _read(replace(facts, grants=(*facts.grants, team)), source)
        case "ReadMember":
            membership: Final = await _read_membership(source) if decision.lookup else source.membership
            member: Final = (
                tuple(membership.litellm_budget_table.allowed_models or ())
                if decision.include and membership is not None and membership.litellm_budget_table is not None
                else ()
            )
            return await _read(replace(facts, grants=(*facts.grants, member)), source)
        case "ReadProject":
            project: Final = object_model_grants(source.project) if source.project is not None else ()
            return await _read(replace(facts, grants=(*facts.grants, project)), source)
        case "ReadOrg":
            org, failed = await _org_grants(decision.id, source)
            return await _read(replace(facts, grants=(*facts.grants, org), org_failed=failed), source)


async def _read_membership(source: _Sources) -> LiteLLM_TeamMembership | None:
    from litellm.proxy.auth.auth_checks import get_team_membership

    if source.team is None or source.token.user_id is None:
        raise NativeAuthRuntimeError("member attribution lookup requested without identity")
    return await get_team_membership(
        user_id=source.token.user_id,
        team_id=source.team.team_id,
        prisma_client=source.prisma,
        user_api_key_cache=source.cache,
        proxy_logging_obj=source.logging,
    )


async def _org_grants(org_id: str | None, source: _Sources) -> tuple[tuple[str, ...], bool]:
    from litellm.proxy.auth.auth_checks import get_org_object

    if org_id is None:
        return (), False
    try:
        org: Final = await get_org_object(
            org_id=org_id,
            prisma_client=source.prisma,
            user_api_key_cache=source.cache,
            proxy_logging_obj=source.logging,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # preserve attribution's org IO failure behavior, never swallow native failure
        verbose_proxy_logger.debug("access group attribution: org lookup failed: %s", error)
        return (), True
    return (tuple(org.models) if org is not None else ()), False


async def collect_matched_model_access_groups(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    project_object: LiteLLM_ProjectTable | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
) -> tuple[str, ...]:
    facts: Final = await read_model_group_attribution_facts(
        model,
        valid_token,
        team_object,
        project_object,
        llm_router,
        prisma_client,
        user_api_key_cache,
        proxy_logging_obj,
        team_membership,
        team_membership_loaded,
    )
    return model_group_attribution(facts).groups


async def stamp_matched_model_access_groups(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    project_object: LiteLLM_ProjectTable | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
) -> tuple[str, ...]:
    try:
        facts: Final = await read_model_group_attribution_facts(
            model,
            valid_token,
            team_object,
            project_object,
            llm_router,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
            team_membership,
            team_membership_loaded,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # native attribution decides the legacy telemetry IO failure result
        verbose_proxy_logger.debug("model access group attribution failed: %s", error)
        return model_group_attribution(ModelGroupAttributionFacts(io_failed=True)).groups
    decision: Final = model_group_attribution(facts)
    if decision.stamp:
        if valid_token is None:
            raise NativeAuthRuntimeError("attribution stamping requested without token")
        valid_token.matched_model_access_groups = list(decision.groups)  # rebind-ok: native attribution effect
    return decision.groups
