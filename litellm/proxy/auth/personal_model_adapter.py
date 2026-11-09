import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

import litellm
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import personal_model_access_applies
from litellm._auth.personal_models import (
    LoadedTeamModelFacts,
    PersonalModelDecision,
    PersonalModelScope,
    loaded_team_models,
    personal_model_scope,
)
from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts
from litellm.proxy.auth.model_projection_adapter import read_grouped_scope
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


def _eligibility(token: UserAPIKeyAuth | None, *, inheritance: bool) -> bool:
    return personal_model_access_applies(
        enabled=bool(litellm.personal_key_model_access_from_teams),
        token_present=token is not None,
        team_id=token.team_id if token is not None else None,
        user_id=token.user_id if token is not None else None,
        master=token.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS if token is not None else False,
        role=token.user_role if token is not None else None,
        api_key_present=token.api_key is not None if token is not None else False,
        inheritance=inheritance,
    )


def personal_key_team_model_access_applies(valid_token: UserAPIKeyAuth | None) -> bool:
    return _eligibility(valid_token, inheritance=False)


def personal_key_inherits_team_grants(valid_token: UserAPIKeyAuth | None) -> bool:
    return _eligibility(valid_token, inheritance=True)


@dataclass(frozen=True, slots=True)
class _Source:
    model: str
    team: LiteLLM_TeamTableCachedObj
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging
    aliases: Mapping[str, str] | None


async def read_personal_model_scope(
    model: str,
    team_object: LiteLLM_TeamTableCachedObj,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> PersonalModelScope:
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check

    aliases: Final = key_model_aliases_for_auth_check(valid_token)
    facts: Final = PersonalModelScope(
        user_present=valid_token.user_id is not None,
        team_id_present=team_object.team_id is not None,  # pyright: ignore[reportUnnecessaryComparison]  # unvalidated cache rows
    )
    return await _read_scope(
        facts,
        _Source(
            model, team_object, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj, aliases
        ),
    )


async def _read_scope(facts: PersonalModelScope, source: _Source) -> PersonalModelScope:
    decision: Final = personal_model_scope(facts)
    if decision.kind == "Complete":
        return facts
    try:
        updated: Final = await _scope_effect(decision, facts, source)
    except NativeAuthFailure:
        raise
    except ProxyException:
        return replace(facts, io_failed=True)
    except Exception as error:  # noqa: BLE001  # preserve personal team lookup failures as raw failure facts, not policy callbacks
        verbose_proxy_logger.warning(
            "Personal key team model access: team=%s check failed, granting nothing: %s", source.team.team_id, error
        )
        return replace(facts, io_failed=True)
    return await _read_scope(updated, source)


async def _scope_effect(
    decision: PersonalModelDecision, facts: PersonalModelScope, source: _Source
) -> PersonalModelScope:
    from litellm.proxy.auth.auth_checks import get_team_membership

    match decision.kind:
        case "ReadTeam":
            team: Final = project_model_facts(
                source.model,
                source.router,
                object_model_grants(source.team),
                team_id=source.team.team_id,
                key_model_aliases=source.aliases,
                object_type="team",
            )
            return replace(facts, team=GroupedModelScope(team, tuple(source.team.access_group_ids or ())))
        case "ReadGroups":
            if facts.team is None:
                raise NativeAuthRuntimeError("personal model group read without team facts")
            grouped: Final = await read_grouped_scope(facts.team, source.router, source.prisma, merge=True)
            return replace(facts, team=grouped)
        case "ReadMembership":
            if source.token.user_id is None:
                raise NativeAuthRuntimeError("personal model membership read without identity")
            membership: Final = await get_team_membership(
                user_id=source.token.user_id,
                team_id=source.team.team_id,
                prisma_client=source.prisma,
                user_api_key_cache=source.cache,
                proxy_logging_obj=source.logging,
            )
            row: Final = membership.litellm_budget_table if membership is not None else None
            grants: Final = tuple(row.allowed_models or ()) if row is not None else ()
            return replace(facts, member_grants=grants)
        case "ReadMember":
            if facts.member_grants is None:
                raise NativeAuthRuntimeError("personal model member scope read without grants")
            member: Final = project_model_facts(
                source.model,
                source.router,
                facts.member_grants,
                team_id=source.team.team_id,
                key_model_aliases=source.aliases,
                object_type="team",
            )
            return replace(facts, member=member)
        case "ReadScopes" | "Complete":
            raise NativeAuthRuntimeError("unexpected personal model scope read")


async def read_loaded_team_model_facts(
    model: str,
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...],
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> LoadedTeamModelFacts:
    facts: Final = LoadedTeamModelFacts(tuple(team is not None for team in teams))
    decision: Final = loaded_team_models(facts)
    if decision.kind == "Complete":
        return facts
    if decision.kind != "ReadScopes":
        raise NativeAuthRuntimeError("unexpected loaded team model read")

    async def read(index: int) -> PersonalModelScope:
        team: Final = teams[index]
        if team is None:
            raise NativeAuthRuntimeError("missing selected personal model team")
        return await read_personal_model_scope(
            model, team, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj
        )

    scopes: Final = tuple(await asyncio.gather(*(read(index) for index in decision.indices)))
    return replace(facts, scopes=scopes)


async def loaded_teams_allow_model(
    model: str,
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...],
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> bool:
    facts: Final = await read_loaded_team_model_facts(
        model, teams, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj
    )
    return loaded_team_models(facts).allowed
