from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.personal_owner import PersonalOwnerFacts, personal_owner
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.owner_team_adapter import read_owner_team_facts
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

if TYPE_CHECKING:
    from litellm.proxy.auth.auth_checks import PersonalKeyOwnerTeams


@dataclass(frozen=True, slots=True)
class PersonalOwnerReading:
    facts: PersonalOwnerFacts
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...] = ()


@dataclass(frozen=True, slots=True)
class _Source:
    token: UserAPIKeyAuth
    prisma: PrismaClient | None
    cache: UserApiKeyCache | None
    logging: ProxyLogging | None
    check_db_only: bool
    owner: LiteLLM_UserTable | None = None


async def read_personal_owner_facts(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
    *,
    check_db_only: bool = False,
) -> PersonalOwnerReading:
    facts: Final = PersonalOwnerFacts(
        valid_token.user_id is not None,
        prisma_client is not None,
        user_api_key_cache is not None,
        proxy_logging_obj is not None,
    )
    return await _read(
        PersonalOwnerReading(facts),
        _Source(valid_token, prisma_client, user_api_key_cache, proxy_logging_obj, check_db_only),
    )


async def personal_key_owner_teams_with_status(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
    *,
    check_db_only: bool = False,
) -> PersonalKeyOwnerTeams:
    from litellm.proxy.auth.auth_checks import PersonalKeyOwnerTeams

    reading: Final = await read_personal_owner_facts(
        valid_token, prisma_client, user_api_key_cache, proxy_logging_obj, check_db_only=check_db_only
    )
    decision: Final = personal_owner(reading.facts)
    if decision.kind != "Complete":
        raise NativeAuthRuntimeError("incomplete personal owner team facts")
    return PersonalKeyOwnerTeams(
        teams=tuple(selected_team(reading.teams, index) for index in decision.selected),
        withheld=tuple(selected_team(reading.teams, index) for index in decision.withheld),
        unreadable=decision.unreadable,
    )


def selected_team(teams: tuple[LiteLLM_TeamTableCachedObj | None, ...], index: int) -> LiteLLM_TeamTableCachedObj:
    team: Final = teams[index]
    if team is None:
        raise NativeAuthRuntimeError("missing selected owner team row")
    return team


async def _read(reading: PersonalOwnerReading, source: _Source) -> PersonalOwnerReading:
    from litellm.proxy.auth.auth_checks import get_user_object

    decision: Final = personal_owner(reading.facts)
    if decision.kind == "Complete":
        return reading
    if source.token.user_id is None or source.prisma is None or source.cache is None or source.logging is None:
        raise NativeAuthRuntimeError("owner read requested without required dependencies")
    match decision.kind:
        case "ReadOwner":
            try:
                owner: Final = await get_user_object(
                    user_id=source.token.user_id,
                    prisma_client=source.prisma,
                    user_api_key_cache=source.cache,
                    user_id_upsert=False,
                    parent_otel_span=source.token.parent_otel_span,
                    proxy_logging_obj=source.logging,
                    check_db_only=source.check_db_only,
                )
            except NativeAuthFailure:
                raise
            except Exception as error:  # noqa: BLE001  # preserve legacy owner lookup failure as raw data
                verbose_proxy_logger.warning(
                    "Personal key team model access: owner lookup failed, granting nothing: %s", error
                )
                return PersonalOwnerReading(replace(reading.facts, owner_loaded=True, owner_failed=True))
            return await _read(
                PersonalOwnerReading(replace(reading.facts, owner_loaded=True, owner_present=owner is not None)),
                replace(source, owner=owner),
            )
        case "ReadTeams":
            if source.owner is None:
                raise NativeAuthRuntimeError("personal owner team read without owner row")
            teams: Final = await read_owner_team_facts(
                source.owner,
                source.token,
                source.prisma,
                source.cache,
                source.logging,
                purpose="Grants",
                check_db_only=source.check_db_only,
            )
            return PersonalOwnerReading(replace(reading.facts, teams=teams.facts), teams.teams)
