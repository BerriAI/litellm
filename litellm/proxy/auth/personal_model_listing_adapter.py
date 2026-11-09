from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.personal_model_listing import PersonalModelListingFacts, personal_model_listing
from litellm._auth.personal_owner import PersonalOwnerFacts, personal_owner
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
from litellm.proxy.auth.personal_model_adapter import read_loaded_team_model_facts
from litellm.proxy.auth.personal_owner_adapter import read_personal_owner_facts, selected_team
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Source:
    names: Sequence[str]
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache | None
    logging: ProxyLogging | None
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...] = ()


async def personal_key_listing_teams(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
    *,
    check_db_only: bool = False,
) -> tuple[LiteLLM_TeamTableCachedObj | None, ...]:
    reading: Final = await read_personal_owner_facts(
        user_api_key_dict,
        prisma_client,
        user_api_key_cache,
        proxy_logging_obj,
        check_db_only=check_db_only,
        purpose="Models",
    )
    decision: Final = personal_owner(reading.facts)
    if decision.kind != "Complete":
        raise NativeAuthRuntimeError("incomplete listing team facts")
    return tuple(team if index in decision.selected else None for index, team in enumerate(reading.teams))


async def read_personal_model_listing_facts(
    models: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
) -> PersonalModelListingFacts:
    owner: Final = PersonalOwnerFacts(
        user_api_key_dict.user_id is not None,
        prisma_client is not None,
        user_api_key_cache is not None,
        proxy_logging_obj is not None,
    )
    return await _read(
        PersonalModelListingFacts(owner),
        _Source(models, user_api_key_dict, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj),
    )


async def personal_key_team_visible_models(
    models: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
) -> tuple[str, ...]:
    facts: Final = await read_personal_model_listing_facts(
        models, user_api_key_dict, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj
    )
    decision: Final = personal_model_listing(facts)
    if decision.kind != "Complete":
        raise NativeAuthRuntimeError("incomplete personal model listing facts")
    return tuple((facts.names or ())[index] for index in decision.selected)


async def _read(facts: PersonalModelListingFacts, source: _Source) -> PersonalModelListingFacts:
    decision: Final = personal_model_listing(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "ReadOwner":
            owner: Final = await read_personal_owner_facts(
                source.token, source.prisma, source.cache, source.logging, purpose="Models"
            )
            return await _read(replace(facts, owner=owner.facts), replace(source, teams=owner.teams))
        case "ReadNames":
            return await _read(replace(facts, names=tuple(source.names)), source)
        case "ReadModel":
            if source.prisma is None or source.cache is None or source.logging is None or facts.names is None:
                raise NativeAuthRuntimeError("personal listing scope read without dependencies")
            teams: Final = tuple(selected_team(source.teams, index) for index in decision.teams)
            model: Final = await read_loaded_team_model_facts(
                facts.names[decision.index],
                teams,
                source.token,
                source.router,
                source.prisma,
                source.cache,
                source.logging,
            )
            return await _read(replace(facts, models=(*facts.models, model)), source)
