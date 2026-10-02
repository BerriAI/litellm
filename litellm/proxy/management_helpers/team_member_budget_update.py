import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Literal

from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.duration_parser import budget_reset_schedule_key
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import LiteLLM_TeamTable as TeamView
from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.management_endpoints.common_utils import (
    _upsert_budget_and_membership,  # pyright: ignore[reportPrivateUsage]  # reuse the field-preserving member budget write
)
from litellm.proxy.management_helpers.access_group_team_sync import TEAM_ADVISORY_LOCK_SQL
from litellm.proxy.utils import PrismaClient
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import TeamMembershipRepository
from litellm.repositories.team_repository import TeamRepository

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma.models import LiteLLM_TeamTable

_METADATA: Final = TypeAdapter(dict[str, JsonValue])
_BUDGET_FIELDS: Final = (
    ("team_member_budget", "max_budget"),
    ("team_member_budget_duration", "budget_duration"),
    ("team_member_rpm_limit", "rpm_limit"),
    ("team_member_tpm_limit", "tpm_limit"),
)


@dataclass(frozen=True, slots=True)
class _TransactionClient:
    db: "Prisma"


@dataclass(frozen=True, slots=True)
class TeamMemberBudgetUpdate:
    team: "LiteLLM_TeamTable"
    budget_id: str
    changed_user_ids: tuple[str, ...]
    member_user_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TeamMemberBudgetUpdateFailure:
    status_code: int
    detail: str


TeamMemberBudgetUpdateResult = TeamMemberBudgetUpdate | TeamMemberBudgetUpdateFailure


def _metadata(value: object) -> dict[str, JsonValue]:
    if value is None:
        return {}
    return _METADATA.validate_json(value) if isinstance(value, str) else _METADATA.validate_python(value)


def matches_member_budget_update(
    *,
    amount: float | None,
    duration: str | None,
    target: float,
    target_duration: str | None,
    mode: Literal["keep", "raise", "lower", "both"],
) -> bool:
    if amount is None or budget_reset_schedule_key(duration) != budget_reset_schedule_key(target_duration):
        return False
    match mode:
        case "raise":
            return amount < target
        case "lower":
            return amount > target
        case "both":
            return amount != target
        case _:
            return False


async def update_team_member_budget_atomic(
    *,
    prisma_client: PrismaClient,
    data: UpdateTeamRequest,
    team_data: Mapping[str, object],
    team_where: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
) -> TeamMemberBudgetUpdateResult:
    async def attempt(remaining: int) -> TeamMemberBudgetUpdateResult:
        try:
            return await _update_team_member_budget_once(
                prisma_client=prisma_client,
                data=data,
                team_data=team_data,
                team_where=team_where,
                user_api_key_dict=user_api_key_dict,
            )
        except Exception as error:
            if remaining == 0 or not PrismaDBExceptionHandler.is_deadlock_error(error):
                raise
            await asyncio.sleep(0.05)
            return await attempt(remaining - 1)

    return await attempt(2)


async def _update_team_member_budget_once(
    *,
    prisma_client: PrismaClient,
    data: UpdateTeamRequest,
    team_data: Mapping[str, object],
    team_where: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
) -> TeamMemberBudgetUpdateResult:
    transaction: Final = prisma_client.tx(timeout=timedelta(seconds=60))
    tx: Final = await transaction.start()
    try:
        result: Final = await _apply_team_member_budget_update(
            tx=tx, data=data, team_data=team_data, team_where=team_where, user_api_key_dict=user_api_key_dict
        )
    except BaseException:
        await transaction.rollback()
        raise
    if isinstance(result, TeamMemberBudgetUpdateFailure):
        await transaction.rollback()
    else:
        await transaction.commit()
    return result


async def lock_team_member_budgets(tx: "Prisma", team_id: str, user_id: str | None = None) -> None:
    await tx.query_raw(TEAM_ADVISORY_LOCK_SQL, team_id)
    await tx.query_raw('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id=$1 FOR UPDATE', team_id)
    await tx.query_raw(
        'SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id=$1 '
        "AND ($2::text IS NULL OR user_id=$2) ORDER BY user_id FOR UPDATE",
        team_id,
        user_id,
    )
    await tx.query_raw(
        'SELECT b.budget_id FROM "LiteLLM_BudgetTable" b WHERE b.budget_id = '
        "(SELECT metadata->>'team_member_budget_id' FROM \"LiteLLM_TeamTable\" WHERE team_id=$1) OR EXISTS "
        '(SELECT 1 FROM "LiteLLM_TeamMembership" m WHERE m.team_id=$1 AND m.budget_id=b.budget_id '
        "AND ($2::text IS NULL OR m.user_id=$2)) ORDER BY b.budget_id FOR UPDATE",
        team_id,
        user_id,
    )


async def _apply_team_member_budget_update(
    *,
    tx: "Prisma",
    data: UpdateTeamRequest,
    team_data: Mapping[str, object],
    team_where: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
) -> TeamMemberBudgetUpdateResult:
    from litellm.proxy.management_helpers.bulk_team_member_budgets import (
        _shared_budget_ids,  # pyright: ignore[reportPrivateUsage]  # preserve budgets shared across memberships
    )

    await lock_team_member_budgets(tx, data.team_id)
    client: Final = _TransactionClient(tx)
    teams: Final = TeamRepository(client).table
    budgets: Final = BudgetRepository(client).table
    memberships: Final = TeamMembershipRepository(client).table
    team_record: Final = await teams.find_unique(where={"team_id": data.team_id})
    if team_record is None:
        return TeamMemberBudgetUpdateFailure(status_code=404, detail="Team not found")
    team: Final = TeamView.model_validate(team_record.model_dump())
    metadata: Final = _metadata(team_record.metadata)
    raw_budget_id: Final = metadata.get("team_member_budget_id")
    budget_id: Final = raw_budget_id if isinstance(raw_budget_id, str) else None
    existing_budget: Final = await budgets.find_unique(where={"budget_id": budget_id}) if budget_id else None
    current_members: Final = await memberships.find_many(
        where={"team_id": data.team_id}, include={"litellm_budget_table": True}
    )
    requested: Final = data.model_dump(exclude_unset=True)
    target_duration: Final = (
        data.team_member_budget_duration
        if "team_member_budget_duration" in data.model_fields_set
        else existing_budget.budget_duration
        if existing_budget is not None
        else team.budget_duration
    )
    reset_fields: Final = (
        {"budget_reset_at": get_budget_reset_time(budget_duration=target_duration) if target_duration else None}
        if existing_budget is None or target_duration != existing_budget.budget_duration
        else {}
    )
    budget_data: Final[Mapping[str, object]] = {
        **{column: requested[field] for field, column in _BUDGET_FIELDS if field in requested},
        "budget_duration": target_duration,
        "updated_by": user_api_key_dict.user_id or "litellm",
        **reset_fields,
    }
    saved_budget: Final = (
        await budgets.update(where={"budget_id": budget_id}, data=budget_data)
        if existing_budget is not None
        else await budgets.create(data={**budget_data, "created_by": user_api_key_dict.user_id or "litellm"})
    )
    if saved_budget is None or saved_budget.max_budget is None:
        return TeamMemberBudgetUpdateFailure(
            status_code=409, detail="Team member budget changed. Reload the team and try again"
        )
    eligible: Final = tuple(
        member
        for member in current_members
        if member.budget_id != saved_budget.budget_id
        and member.litellm_budget_table is not None
        and matches_member_budget_update(
            amount=member.litellm_budget_table.max_budget,
            duration=member.litellm_budget_table.budget_duration,
            target=saved_budget.max_budget,
            target_duration=target_duration,
            mode=data.team_member_budget_update_mode or "keep",
        )
    )
    shared_ids: Final = await _shared_budget_ids(
        tx, frozenset(member.budget_id for member in eligible if member.budget_id is not None)
    )
    for member in eligible:
        await _upsert_budget_and_membership(
            tx,
            team_id=data.team_id,
            user_id=member.user_id,
            existing_budget_id=member.budget_id,
            user_api_key_dict=user_api_key_dict,
            budget_patch={"max_budget": None},
            team_default_budget_id=saved_budget.budget_id,
            shared_budget_ids=shared_ids,
        )
    existing_ids: Final = frozenset(member.user_id for member in current_members)
    roster_ids: Final = tuple(
        member.user_id for member in (team.members_with_roles or ()) if member.user_id is not None
    )
    missing: Final = tuple(
        {"team_id": data.team_id, "user_id": user_id, "budget_id": saved_budget.budget_id}
        for user_id in roster_ids
        if user_id not in existing_ids
    )
    if missing:
        await memberships.create_many(data=missing, skip_duplicates=True)
    await memberships.update_many(
        where={"team_id": data.team_id, "budget_id": None, "user_id": {"not_in": [m.user_id for m in eligible]}},
        data={"budget_id": saved_budget.budget_id},
    )
    written: Final = await teams.update_many(
        where=team_where,
        data={
            **team_data,
            "metadata": safe_dumps(
                {**_metadata(team_data.get("metadata", metadata)), "team_member_budget_id": saved_budget.budget_id}
            ),
        },
    )
    if written != 1:
        return TeamMemberBudgetUpdateFailure(
            status_code=409, detail="Team budget changed. Reload the team and try again"
        )
    updated_team: Final = await teams.find_unique(
        where={"team_id": data.team_id}, include={"litellm_model_table": True, "object_permission": True}
    )
    if updated_team is None:
        return TeamMemberBudgetUpdateFailure(status_code=409, detail="Team changed. Reload the team and try again")
    return TeamMemberBudgetUpdate(
        team=updated_team,
        budget_id=saved_budget.budget_id,
        changed_user_ids=tuple(member.user_id for member in eligible),
        member_user_ids=tuple(sorted(existing_ids | frozenset(roster_ids))),
    )
