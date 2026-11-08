from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.member_models import MemberModelFacts, member_models
from litellm.proxy._types import LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import project_model_facts, raise_model_denial
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Source:
    model: str | Sequence[str]
    team: LiteLLM_TeamTable
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging
    membership: LiteLLM_TeamMembership | None
    aliases: Mapping[str, str] | None


async def read_member_model_facts(
    model: str | Sequence[str],
    team_object: LiteLLM_TeamTable,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
    key_model_aliases: Mapping[str, str] | None = None,
) -> MemberModelFacts:
    facts: Final = MemberModelFacts(
        user_present=valid_token.user_id is not None,
        team_present=team_object.team_id is not None,  # pyright: ignore[reportUnnecessaryComparison]  # preserve unvalidated cache row identity guard
        membership_loaded=team_membership_loaded,
    )
    return await _read(
        facts,
        _Source(
            model,
            team_object,
            valid_token,
            llm_router,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
            team_membership,
            key_model_aliases,
        ),
    )


async def check_team_member_model_access(
    model: str | Sequence[str],
    team_object: LiteLLM_TeamTable,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
    key_model_aliases: Mapping[str, str] | None = None,
) -> None:
    facts: Final = await read_member_model_facts(
        model,
        team_object,
        valid_token,
        llm_router,
        prisma_client,
        user_api_key_cache,
        proxy_logging_obj,
        team_membership,
        team_membership_loaded,
        key_model_aliases,
    )
    decision: Final = member_models(facts)
    if decision.kind == "Allow":
        return
    if decision.kind != "Deny" or decision.denial is None:
        raise NativeAuthRuntimeError("incomplete member model access facts")
    raise_model_denial(
        decision.denial,
        model,
        user_id=valid_token.user_id,
        team_id=team_object.team_id,
        member_models=decision.member_models,
    )


async def _read(facts: MemberModelFacts, source: _Source) -> MemberModelFacts:
    from litellm.proxy.auth.auth_checks import get_team_membership

    decision: Final = member_models(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadMembership":
            if source.token.user_id is None:
                raise NativeAuthRuntimeError("member model lookup without user identity")
            membership: Final = await get_team_membership(
                user_id=source.token.user_id,
                team_id=source.team.team_id,
                prisma_client=source.prisma,
                user_api_key_cache=source.cache,
                proxy_logging_obj=source.logging,
            )
            return await _read(replace(facts, membership_loaded=True), replace(source, membership=membership))
        case "ReadGrants":
            row: Final = source.membership.litellm_budget_table if source.membership is not None else None
            return await _read(
                replace(facts, grants=tuple(row.allowed_models or ()) if row is not None else ()), source
            )
        case "ReadScope":
            if facts.grants is None:
                raise NativeAuthRuntimeError("member model scope read without grants")
            scope: Final = project_model_facts(
                source.model,
                source.router,
                facts.grants,
                team_id=source.team.team_id,
                key_model_aliases=source.aliases,
                object_type="team",
            )
            return await _read(replace(facts, scope=scope), source)
