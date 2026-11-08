from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel

import litellm
from litellm._auth.common_budgets import BudgetFacts, MemberLimitFacts, WindowFacts, window_start
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.personal_budget import PersonalTeamBudgetDecision, PersonalTeamBudgetFacts, personal_team_budget
from litellm._logging import verbose_proxy_logger
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import Litellm_EntityType, LiteLLM_TeamMembership, LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
from litellm.proxy.auth.budget_alert_adapter import team_member_max_budget_alert_check
from litellm.proxy.auth.budget_windows_adapter import (
    _reset,  # pyright: ignore[reportPrivateUsage]  # shared raw window timestamp projection
    _seconds,  # pyright: ignore[reportPrivateUsage]  # shared raw duration projection
    _window,  # pyright: ignore[reportPrivateUsage]  # shared typed window decoding
)
from litellm.proxy.auth.common_budget_adapter import CounterRead, emit_alert, key_alert_info, metadata, read_budget
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.carried_budget_state import carry_organization_budget_state
from litellm.proxy.utils import PrismaClient, ProxyLogging


class _Windows(BaseModel, frozen=True):
    budget_limits: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class _Sources:
    team: LiteLLM_TeamTableCachedObj
    token: UserAPIKeyAuth
    prisma: PrismaClient
    cache: UserApiKeyCache
    logging: ProxyLogging
    windows: tuple[object, ...] = ()
    membership: LiteLLM_TeamMembership | None = None


async def read_personal_team_budget_facts(
    team: LiteLLM_TeamTableCachedObj,
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    *,
    check_db_only: bool = False,
) -> PersonalTeamBudgetFacts:
    facts: Final = PersonalTeamBudgetFacts(
        user_present=valid_token.user_id is not None,
        team_id_present=team.team_id is not None,  # pyright: ignore[reportUnnecessaryComparison]  # unvalidated cache rows
        check_db_only=check_db_only,
        token_org=valid_token.org_id,
        team_org=team.organization_id,
    )
    decision: Final = personal_team_budget(facts)
    if decision.kind != "Source":
        raise NativeAuthRuntimeError("personal budget source was not selected")
    source: Final = valid_token.model_copy(update={"team_id": team.team_id, "org_id": decision.org_id})
    return await _read(
        replace(facts, source_ready=True), _Sources(team, source, prisma_client, user_api_key_cache, proxy_logging_obj)
    )


async def personal_key_team_within_budget(
    team: LiteLLM_TeamTableCachedObj,
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    *,
    check_db_only: bool = False,
) -> bool:
    facts: Final = await read_personal_team_budget_facts(
        team, valid_token, prisma_client, user_api_key_cache, proxy_logging_obj, check_db_only=check_db_only
    )
    return personal_team_budget(facts).allowed


async def _read(facts: PersonalTeamBudgetFacts, source: _Sources) -> PersonalTeamBudgetFacts:
    decision: Final = personal_team_budget(facts)
    if decision.kind == "Complete":
        if decision.log_failure:
            verbose_proxy_logger.warning(
                "Failed to check budget for personal key team %s, skipping: incompatible datetimes", source.team.team_id
            )
        return facts
    try:
        updated, updated_source = await _effect(decision, facts, source)
    except NativeAuthFailure:
        raise
    except litellm.BudgetExceededError:
        return replace(facts, io_failed=True)
    except Exception as error:  # noqa: BLE001  # preserve fail-closed personal budget IO behavior without catching native failures
        verbose_proxy_logger.warning(
            "Failed to check budget for personal key team %s, skipping: %s", source.team.team_id, error
        )
        return replace(facts, io_failed=True)
    return await _read(updated, updated_source)


def _member_limit(row: LiteLLM_BudgetTable | None, now: datetime) -> MemberLimitFacts:
    expiry: Final = row.temp_budget_expiry if row is not None else None
    aware: Final = expiry.replace(tzinfo=timezone.utc) if expiry is not None and expiry.tzinfo is None else expiry
    expiry_number: Final = (
        aware.replace(tzinfo=timezone.utc).timestamp()
        if aware is not None and aware.tzinfo is now.tzinfo
        else aware.timestamp()
        if aware is not None
        else None
    )
    now_number: Final = (
        now.replace(tzinfo=timezone.utc).timestamp()
        if aware is not None and aware.tzinfo is now.tzinfo
        else now.timestamp()
    )
    return MemberLimitFacts(
        row.max_budget if row is not None else None,
        None,
        row.temp_budget_increase if row is not None else None,
        expiry_number,
        now_number,
        now.utcoffset() is None,
    )


async def _effect(
    decision: PersonalTeamBudgetDecision, facts: PersonalTeamBudgetFacts, source: _Sources
) -> tuple[PersonalTeamBudgetFacts, _Sources]:
    from litellm.proxy.auth.auth_checks import get_team_member_default_budget, get_team_membership, get_utc_datetime

    team: Final = source.team
    match decision.kind:
        case "ReadTeam":
            budget, _decision = await read_budget(
                BudgetFacts("Team", team.max_budget, token_present=True),
                CounterRead(f"spend:team:{team.team_id}", team.spend or 0.0, team.max_budget),
            )
            return replace(facts, team=budget), source
        case "AlertTeam":
            if facts.team is None:
                raise NativeAuthRuntimeError("missing personal team budget alert facts")
            emit_alert(
                source.logging,
                "team_budget",
                key_alert_info(source.token, facts.team.spend, facts.team.limit, Litellm_EntityType.TEAM),
            )
            return replace(facts, team_alerted=True), source
        case "ReadWindows":
            windows: Final = _Windows.model_validate(team, from_attributes=True).budget_limits or ()
            return replace(facts, window_count=len(windows)), replace(source, windows=windows)
        case "ReadWindow":
            window: Final = _window(source.windows[decision.index])
            seconds: Final = _seconds(window.budget_duration)
            reset: Final = _reset(window.reset_at)
            start: Final = window_start(
                WindowFacts(
                    seconds, reset.timestamp() if reset is not None else None, datetime.now(timezone.utc).timestamp()
                )
            )
            window_facts, _window_decision = await read_budget(
                BudgetFacts("TeamWindow", window.max_budget),
                CounterRead(
                    f"spend:team:{team.team_id}:window:{window.budget_duration}",
                    0.0,
                    window.max_budget,
                    window_entity_type="Team",
                    window_entity_id=team.team_id,
                    window_duration=window.budget_duration,
                    window_start=datetime.fromtimestamp(start, timezone.utc) if start is not None else None,
                ),
            )
            return replace(facts, windows=(*facts.windows, window_facts)), source
        case "ReadMembership":
            if source.token.user_id is None:
                raise NativeAuthRuntimeError("missing personal team membership identity")
            membership: Final = await get_team_membership(
                user_id=source.token.user_id,
                team_id=team.team_id,
                prisma_client=source.prisma,
                user_api_key_cache=source.cache,
                proxy_logging_obj=source.logging,
                check_db_only=facts.check_db_only,
            )
            return replace(facts, membership_loaded=True), replace(source, membership=membership)
        case "ReadMemberLimit":
            row: Final = source.membership.litellm_budget_table if source.membership is not None else None
            now: Final = get_utc_datetime()
            default_id: Final = metadata(team.metadata).get("team_member_budget_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy metadata projected as an opaque mapping
            return replace(
                facts,
                member_limit=_member_limit(row, now),
                default_id=default_id if isinstance(default_id, str) else None,
            ), source
        case "ReadDefault":
            if facts.member_limit is None:
                raise NativeAuthRuntimeError("missing personal member limit facts")
            default: Final = await get_team_member_default_budget(
                budget_id=decision.id, prisma_client=source.prisma, user_api_key_cache=source.cache
            )
            return replace(
                facts,
                default_loaded=True,
                member_limit=replace(
                    facts.member_limit, default_limit=default.max_budget if default is not None else None
                ),
            ), source
        case "ReadMember":
            member, _member_decision = await read_budget(
                BudgetFacts("TeamMember", decision.limit),
                CounterRead(
                    f"spend:team_member:{source.token.user_id}:{team.team_id}",
                    (source.membership.spend or 0.0) if source.membership is not None else 0.0,
                    decision.limit,
                ),
            )
            return replace(facts, member=member), source
        case "AlertMember":
            alert_member: Final = facts.member
            if (
                alert_member is None
                or alert_member.spend is None
                or alert_member.limit is None
                or source.token.user_id is None
            ):
                raise NativeAuthRuntimeError("missing personal member alert facts")
            team_member_max_budget_alert_check(
                team_id=team.team_id,
                team_alias=team.team_alias,
                team_metadata=team.metadata,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # helper validates legacy alert metadata
                organization_id=team.organization_id,
                user_id=source.token.user_id,
                user_email=None,
                proxy_logging_obj=source.logging,
                spend=alert_member.spend,
                max_budget=alert_member.limit,
            )
            return replace(facts, member_alerted=True), source
        case "ReadOrg":
            org_budget, org_failed = await _read_org(decision.id, source)
            return replace(facts, org_loaded=True, org=org_budget, org_failed=org_failed), source
        case "AlertOrg":
            if facts.org is None:
                raise NativeAuthRuntimeError("missing personal org budget alert facts")
            emit_alert(
                source.logging,
                "organization_budget",
                key_alert_info(
                    source.token, facts.org.spend, facts.org.limit, Litellm_EntityType.ORGANIZATION
                ).model_copy(update={"organization_id": decision.id}),
            )
            return replace(facts, org_alerted=True), source
        case "Source" | "Complete":
            raise NativeAuthRuntimeError("unexpected personal budget read")


async def _read_org(org_id: str, source: _Sources) -> tuple[BudgetFacts | None, bool]:
    from litellm.proxy.auth.auth_checks import get_org_object

    try:
        org: Final = await get_org_object(
            org_id=org_id,
            prisma_client=source.prisma,
            user_api_key_cache=source.cache,
            proxy_logging_obj=source.logging,
            include_budget_table=True,
        )
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # legacy org lookup failure skips only the org budget, not the other personal-team ceilings
        return None, True
    if org is None:
        return None, False
    carry_organization_budget_state(valid_token=source.token, org_table=org)
    limit: Final = org.litellm_budget_table.max_budget if org.litellm_budget_table is not None else None
    budget, _decision = await read_budget(
        BudgetFacts("Organization", limit), CounterRead(f"spend:org:{org_id}", org.spend or 0.0, limit)
    )
    return budget, False
