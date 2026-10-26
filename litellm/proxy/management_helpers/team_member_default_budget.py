import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.litellm_core_utils.duration_parser import budget_reset_period_key
from litellm.proxy._types import (
    TeamMemberBudgetBackfill,
    TeamMemberBudgetBackfillResult,
    UpdateTeamRequest,
)
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE: Final = 50_000

_DEFAULT_BUDGET_COLUMNS: Final = MappingProxyType(
    {
        "team_member_budget": "max_budget",
        "team_member_rpm_limit": "rpm_limit",
        "team_member_tpm_limit": "tpm_limit",
        "team_member_budget_duration": "budget_duration",
        "team_member_model_max_budget": "model_max_budget",
    }
)

_LINK_MISSING_MEMBERSHIPS_SQL: Final = """
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id)
SELECT member.user_id, $1, $2 FROM unnest($3::text[]) AS member(user_id)
ON CONFLICT (user_id, team_id) DO NOTHING
"""
_LINK_UNBUDGETED_MEMBERSHIPS_SQL: Final = (
    'UPDATE "LiteLLM_TeamMembership" SET budget_id = $2 WHERE team_id = $1 AND budget_id IS NULL'
)
_ON_CHOSEN_SIDE_OF_DEFAULT: Final = """m.team_id = $1 AND m.budget_id <> $2 AND CASE $3::text
    WHEN 'raise_to_default' THEN b.max_budget < $4::double precision
    WHEN 'lower_to_default' THEN b.max_budget > $4::double precision
    WHEN 'all_to_default' THEN b.max_budget <> $4::double precision
    ELSE FALSE END"""
_CANDIDATES_BY_SCHEDULE_SQL: Final = (
    'SELECT b.budget_duration AS duration, COUNT(*)::int AS members FROM "LiteLLM_TeamMembership" m '
    'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id WHERE '
    + _ON_CHOSEN_SIDE_OF_DEFAULT
    + " GROUP BY b.budget_duration"
)
_MOVE_MEMBERS_SQL: Final = (
    """
WITH candidates AS MATERIALIZED (
    SELECT m.user_id, b.*, CASE WHEN
        b.soft_budget IS NOT NULL OR b.max_parallel_requests IS NOT NULL OR
        b.tpm_limit IS NOT NULL OR b.rpm_limit IS NOT NULL OR b.tpd_limit IS NOT NULL OR
        b.temp_budget_increase IS NOT NULL OR b.temp_budget_expiry IS NOT NULL OR
        COALESCE(b.model_max_budget, '{}'::jsonb) NOT IN ('{}'::jsonb, 'null'::jsonb) OR
        cardinality(b.allowed_models) > 0
    THEN gen_random_uuid()::text ELSE $2 END AS next_budget_id
    FROM "LiteLLM_TeamMembership" m JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id
    WHERE """
    + _ON_CHOSEN_SIDE_OF_DEFAULT
    + """ AND (b.budget_duration = ANY($5::text[]) OR ($6::boolean AND b.budget_duration IS NULL))
), copies AS (
    INSERT INTO "LiteLLM_BudgetTable" (
        budget_id, max_budget, soft_budget, max_parallel_requests, tpm_limit, rpm_limit, tpd_limit,
        model_max_budget, budget_duration, budget_reset_at, allowed_models, temp_budget_increase,
        temp_budget_expiry, created_by, updated_by, created_at, updated_at
    )
    SELECT next_budget_id, NULL, soft_budget, max_parallel_requests, tpm_limit, rpm_limit, tpd_limit,
        model_max_budget, budget_duration,
        (SELECT budget_reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = $2),
        allowed_models, temp_budget_increase, temp_budget_expiry, $7, $7,
        NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC'
    FROM candidates WHERE next_budget_id <> $2
    RETURNING budget_id
)
UPDATE "LiteLLM_TeamMembership" m SET budget_id = c.next_budget_id
FROM candidates c
WHERE m.team_id = $1 AND m.user_id = c.user_id
    AND (c.next_budget_id = $2 OR c.next_budget_id IN (SELECT budget_id FROM copies))
RETURNING m.user_id
"""
)


@dataclass(frozen=True, slots=True)
class TeamMemberDefaultBudgetWrite:
    budget_id: str
    update: Mapping[str, object]
    create: Mapping[str, object]
    links_members: bool


@dataclass(frozen=True, slots=True)
class MovedMemberBudgets:
    user_ids: tuple[str, ...]
    skipped_other_reset_period: int

    def result(self) -> TeamMemberBudgetBackfillResult:
        return TeamMemberBudgetBackfillResult(
            moved_to_default=len(self.user_ids), skipped_other_reset_period=self.skipped_other_reset_period
        )


@dataclass(frozen=True, slots=True)
class TooManyMemberBudgetsToMove:
    count: int


class _ScheduleCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration: str | None
    members: int


class _MovedMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SCHEDULE_COUNTS: Final = TypeAdapter(tuple[_ScheduleCount, ...])
_MOVED_MEMBERS: Final = TypeAdapter(tuple[_MovedMember, ...])


def plan_team_member_default_budget_write(
    data: UpdateTeamRequest,
    existing_budget_id: str | None,
    team_alias: str | None,
    team_budget_duration: str | None,
) -> TeamMemberDefaultBudgetWrite | None:
    requested: Final = _JSON_OBJECT.validate_python(
        data.model_dump(mode="json", include=data.model_fields_set & _DEFAULT_BUDGET_COLUMNS.keys())
    )
    sent: Final = MappingProxyType({_DEFAULT_BUDGET_COLUMNS[field]: value for field, value in requested.items()})
    links_members: Final = any(value is not None for value in sent.values())
    if not sent or (existing_budget_id is None and not links_members):
        return None
    create_duration: Final = sent.get("budget_duration", team_budget_duration)
    created: Final = {**sent, "budget_duration": create_duration}
    return TeamMemberDefaultBudgetWrite(
        budget_id=existing_budget_id or _new_default_budget_id(team_alias),
        update=_with_reset_at(
            {column: {} if column == "model_max_budget" and value is None else value for column, value in sent.items()}
        ),
        create=_with_reset_at({column: value for column, value in created.items() if value is not None}),
        links_members=links_members,
    )


async def write_team_member_default_budget(
    tx: "Prisma",
    write: TeamMemberDefaultBudgetWrite,
    *,
    team_id: str,
    member_user_ids: Sequence[str],
    changed_by: str,
) -> None:
    """The caller holds the team advisory lock on ``tx``, so the members linked here are the whole roster."""
    stamps: Final = {"updated_by": changed_by}
    await _budget_tx_db(tx).upsert(
        where={"budget_id": write.budget_id},
        data={
            "create": _prisma_columns(
                {**write.create, **stamps, "budget_id": write.budget_id, "created_by": changed_by}
            ),
            "update": _prisma_columns({**write.update, **stamps}),
        },
    )
    if not write.links_members:
        return
    await tx.execute_raw(_LINK_MISSING_MEMBERSHIPS_SQL, team_id, write.budget_id, list(member_user_ids))
    await tx.execute_raw(_LINK_UNBUDGETED_MEMBERSHIPS_SQL, team_id, write.budget_id)


async def move_member_budgets_to_default(
    tx: "Prisma",
    *,
    team_id: str,
    default_budget_id: str,
    backfill: TeamMemberBudgetBackfill,
    changed_by: str,
) -> MovedMemberBudgets | TooManyMemberBudgetsToMove:
    if backfill == "none":
        return MovedMemberBudgets(user_ids=(), skipped_other_reset_period=0)
    default: Final = await _budget_tx_db(tx).find_unique(where={"budget_id": default_budget_id})
    if default is None or default.max_budget is None:
        return MovedMemberBudgets(user_ids=(), skipped_other_reset_period=0)
    schedule: Final = budget_reset_period_key(default.budget_duration)
    counts: Final = _SCHEDULE_COUNTS.validate_python(
        await tx.query_raw(_CANDIDATES_BY_SCHEDULE_SQL, team_id, default_budget_id, backfill, default.max_budget)
    )
    matching: Final = tuple(count for count in counts if budget_reset_period_key(count.duration) == schedule)
    moving: Final = sum(count.members for count in matching)
    skipped: Final = sum(count.members for count in counts) - moving
    if moving > MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE:
        return TooManyMemberBudgetsToMove(count=moving)
    if moving == 0:
        return MovedMemberBudgets(user_ids=(), skipped_other_reset_period=skipped)
    moved: Final = _MOVED_MEMBERS.validate_python(
        await tx.query_raw(
            _MOVE_MEMBERS_SQL,
            team_id,
            default_budget_id,
            backfill,
            default.max_budget,
            [count.duration for count in matching if count.duration is not None],
            any(count.duration is None for count in matching),
            changed_by,
        )
    )
    return MovedMemberBudgets(user_ids=tuple(member.user_id for member in moved), skipped_other_reset_period=skipped)


def _prisma_columns(columns: Mapping[str, object]) -> Mapping[str, object]:
    from prisma import Json  # noqa: PLC0415  # local import: prisma may be ungenerated at module load in some tools

    return {column: Json(value) if isinstance(value, dict) else value for column, value in columns.items()}


def _budget_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_BudgetTable]":
    return tx.litellm_budgettable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _with_reset_at(columns: Mapping[str, object]) -> Mapping[str, object]:
    if "budget_duration" not in columns:
        return MappingProxyType(dict(columns))
    duration: Final = columns["budget_duration"]
    reset_at: Final = get_budget_reset_time(budget_duration=duration) if isinstance(duration, str) else None
    return MappingProxyType({**columns, "budget_reset_at": reset_at})


def _new_default_budget_id(team_alias: str | None) -> str:
    if team_alias is None:
        return f"team-budget-{uuid.uuid4().hex}"
    return f"team-{team_alias.replace(' ', '-')}-budget-{uuid.uuid4().hex}"
