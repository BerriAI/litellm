import asyncio
from dataclasses import dataclass, replace
from typing import Final, Literal

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.owner_teams import OwnerTeamBudget, OwnerTeamFacts, OwnerTeamLoad, owner_teams
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UI_TEAM_ID, LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.personal_budget_adapter import read_personal_team_budget_facts
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging


@dataclass(frozen=True, slots=True)
class OwnerTeamReading:
    facts: OwnerTeamFacts
    teams: tuple[LiteLLM_TeamTableCachedObj | None, ...] = ()


@dataclass(frozen=True, slots=True)
class _Source:
    token: UserAPIKeyAuth
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging
    check_db_only: bool


async def read_owner_team_facts(
    user_object: LiteLLM_UserTable,
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    *,
    purpose: Literal["Models", "Grants"],
    check_db_only: bool = False,
) -> OwnerTeamReading:
    facts: Final = OwnerTeamFacts(
        purpose,
        tuple(user_object.teams),
        UI_TEAM_ID,
        user_present=valid_token.user_id is not None,
        database_present=prisma_client is not None,
        org=valid_token.org_id,
    )
    return await _read(
        OwnerTeamReading(facts),
        _Source(valid_token, prisma_client, user_api_key_cache, proxy_logging_obj, check_db_only),
    )


async def _load(team_id: str, source: _Source) -> tuple[OwnerTeamLoad, LiteLLM_TeamTableCachedObj | None]:
    from litellm.proxy.auth.auth_checks import get_team_object

    try:
        team: Final = await get_team_object(
            team_id=team_id,
            prisma_client=source.prisma,
            user_api_key_cache=source.cache,
            parent_otel_span=source.token.parent_otel_span,
            proxy_logging_obj=source.logging,
            check_db_only=source.check_db_only,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # raw team lookup failures grant nothing; native failure is never converted
        verbose_proxy_logger.warning(
            "Personal key team model access: team=%s lookup failed, granting nothing: %s", team_id, error
        )
        return OwnerTeamLoad(False, failed=True), None
    return OwnerTeamLoad(True, blocked=team.blocked is True, org=team.organization_id), team


async def _budget(index: int, reading: OwnerTeamReading, source: _Source) -> OwnerTeamBudget:
    team: Final = reading.teams[index]
    if team is None or source.prisma is None:
        raise NativeAuthRuntimeError("owner team budget read requested without its source")
    return OwnerTeamBudget(
        index,
        await read_personal_team_budget_facts(
            team, source.token, source.prisma, source.cache, source.logging, check_db_only=source.check_db_only
        ),
    )


async def _read(reading: OwnerTeamReading, source: _Source) -> OwnerTeamReading:
    decision: Final = owner_teams(reading.facts)
    match decision.kind:
        case "Complete":
            return reading
        case "ReadTeams":
            rows: Final = await asyncio.gather(*(_load(team_id, source) for team_id in decision.ids))
            updated: Final = OwnerTeamReading(
                replace(reading.facts, loaded=True, loads=tuple(row[0] for row in rows)), tuple(row[1] for row in rows)
            )
            return await _read(updated, source)
        case "ReadBudgets":
            budgets: Final = tuple(
                await asyncio.gather(*(_budget(index, reading, source) for index in decision.indices))
            )
            return await _read(
                replace(reading, facts=replace(reading.facts, budgets=(*reading.facts.budgets, *budgets))), source
            )
