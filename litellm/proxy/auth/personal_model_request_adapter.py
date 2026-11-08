from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.model_access import ModelAccessDecision
from litellm._auth.personal_model_request import PersonalModelRequestFacts, personal_model_request
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import raise_model_denial
from litellm.proxy.auth.owner_team_adapter import read_owner_team_facts
from litellm.proxy.auth.personal_model_adapter import read_loaded_team_model_facts
from litellm.proxy.auth.personal_owner_adapter import selected_team
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Source:
    owner: LiteLLM_UserTable | None
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...] = ()


async def read_personal_model_request_facts(
    model: str | Sequence[str],
    user_object: LiteLLM_UserTable | None,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> PersonalModelRequestFacts:
    requested: Final = (model,) if isinstance(model, str) else tuple(model)
    return await _read(
        PersonalModelRequestFacts(requested, user_object is not None),
        _Source(user_object, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj),
    )


async def can_personal_key_call_model_via_teams(
    model: str | Sequence[str],
    user_object: LiteLLM_UserTable | None,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> Literal[True]:
    facts: Final = await read_personal_model_request_facts(
        model, user_object, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj
    )
    decision: Final = personal_model_request(facts)
    if decision.kind == "Allow":
        return True
    if decision.kind != "Deny":
        raise NativeAuthRuntimeError("incomplete personal model request facts")
    return raise_model_denial(
        ModelAccessDecision("PersonalTeam", model=decision.model),
        model,
        user_id=valid_token.user_id,
        teams=user_object.teams if user_object is not None else None,
    )


async def _read(facts: PersonalModelRequestFacts, source: _Source) -> PersonalModelRequestFacts:
    decision: Final = personal_model_request(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadTeams":
            if source.owner is None:
                raise NativeAuthRuntimeError("personal model team read without owner")
            reading: Final = await read_owner_team_facts(
                source.owner, source.token, source.prisma, source.cache, source.logging, purpose="Models"
            )
            return await _read(replace(facts, teams=reading.facts), replace(source, teams=reading.teams))
        case "ReadModel":
            teams: Final = tuple(selected_team(source.teams, index) for index in decision.teams)
            model: Final = await read_loaded_team_model_facts(
                facts.requested[decision.index],
                teams,
                source.token,
                source.router,
                source.prisma,
                source.cache,
                source.logging,
            )
            return await _read(replace(facts, models=(*facts.models, model)), source)
