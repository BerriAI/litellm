from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, assert_never

from litellm._auth import budget_lookup as core
from litellm._auth.failure import NativeAuthFailure
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.common_budget_adapter import metadata
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging


@dataclass(frozen=True, slots=True)
class LookupIO:
    token: UserAPIKeyAuth | None
    team: LiteLLM_TeamTable | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging


@dataclass(frozen=True, slots=True)
class OrganizationReading:
    org_id: str
    row: LiteLLM_OrganizationTable


async def organization(
    facts: core.OrganizationFacts, io: LookupIO, org_id: str | None = None
) -> OrganizationReading | None:
    from litellm.proxy.auth.auth_checks import get_org_object

    action: Final = core.organization(facts)
    match action:
        case "Skip":
            return None
        case "ReadTokenId":
            assert io.token is not None
            token_id: Final = io.token.org_id
            return await organization(replace(facts, token_id=token_id is not None), io, token_id)
        case "ReadTeamId":
            assert io.team is not None
            team_id: Final = io.team.organization_id
            return await organization(replace(facts, team_id=team_id is not None), io, team_id)
        case "ReadOrganization":
            assert org_id is not None and io.prisma is not None
            try:
                row: Final = await get_org_object(
                    org_id=org_id,
                    prisma_client=io.prisma,
                    user_api_key_cache=io.cache,
                    proxy_logging_obj=io.logging,
                    include_budget_table=True,
                )
            except NativeAuthFailure:
                raise
            except Exception:  # noqa: BLE001  # pass the raw lookup failure to native Phase 1 policy
                return await organization(replace(facts, lookup_failed=True), io, org_id)
            result: Final = core.organization(replace(facts, row_present=row is not None))
            assert result in ("Skip", "UseOrganization")
            if result == "Skip":
                return None
            assert row is not None
            return OrganizationReading(org_id, row)
        case "UseOrganization":
            raise AssertionError("organization row was not retained")
        case _:
            assert_never(action)


@dataclass(frozen=True, slots=True)
class MemberReading:
    membership: LiteLLM_TeamMembership | None
    row: LiteLLM_BudgetTable | None
    now: datetime
    metadata: Mapping[str, object]
    default_id: object
    default: LiteLLM_BudgetTable | None = None


async def member(
    facts: core.MemberFacts,
    io: LookupIO,
    membership: LiteLLM_TeamMembership | None,
    reading: MemberReading | None = None,
) -> MemberReading | None:
    from litellm.proxy.auth.auth_checks import get_team_member_default_budget, get_team_membership, get_utc_datetime

    action: Final = core.member(facts)
    match action:
        case "Skip":
            return None
        case "ReadTeamId":
            assert io.team is not None
            return await member(replace(facts, team_id=io.team.team_id is not None), io, membership)  # pyright: ignore[reportUnnecessaryComparison]  # Phase 1 accepts constructed cache rows with missing IDs
        case "ReadUserId":
            assert io.token is not None
            return await member(replace(facts, user_id=io.token.user_id is not None), io, membership)
        case "ReadMembership":
            assert io.token is not None and io.token.user_id is not None and io.team is not None
            loaded: Final = await get_team_membership(
                user_id=io.token.user_id,
                team_id=io.team.team_id,
                prisma_client=io.prisma,
                user_api_key_cache=io.cache,
                proxy_logging_obj=io.logging,
            )
            return await member(replace(facts, membership_loaded=True), io, loaded)
        case "ReadLimits":
            assert io.team is not None
            row: Final = membership.litellm_budget_table if membership is not None else None
            now: Final = get_utc_datetime()
            values: Final = metadata(io.team.metadata)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # raw legacy metadata boundary
            default_id: Final = values.get("team_member_budget_id")
            snapshot: Final = MemberReading(membership, row, now, values, default_id)
            return await member(
                replace(
                    facts,
                    limits_read=True,
                    maximum_present=row is not None and row.max_budget is not None,
                    default_id_string=isinstance(default_id, str),
                ),
                io,
                membership,
                snapshot,
            )
        case "ReadDefault":
            assert reading is not None and isinstance(reading.default_id, str)
            default: Final = await get_team_member_default_budget(
                budget_id=reading.default_id, prisma_client=io.prisma, user_api_key_cache=io.cache
            )
            return await member(replace(facts, default_read=True), io, membership, replace(reading, default=default))
        case "UseLimits":
            assert reading is not None
            return reading
        case _:
            assert_never(action)
