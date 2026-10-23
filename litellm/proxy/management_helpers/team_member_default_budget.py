import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, field_validator

from litellm.litellm_core_utils.duration_parser import budget_reset_period_key
from litellm.proxy._types import (
    TeamMemberBudgetBackfill,
    TeamMemberBudgetBackfillResult,
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.common_utils import (
    has_meaningful_budget_limit,
    upsert_budget_and_membership,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE: Final = 50_000

_ALWAYS_OWN_LIMITS: Final = frozenset({"allowed_models"})

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
_RELINK_TO_DEFAULT_SQL: Final = (
    'UPDATE "LiteLLM_TeamMembership" SET budget_id = $2 WHERE team_id = $1 AND user_id = ANY($3::text[])'
)
_CLEAR_MAX_BUDGET_SQL: Final = """
UPDATE "LiteLLM_BudgetTable" SET max_budget = NULL, updated_by = $2, updated_at = (NOW() AT TIME ZONE 'UTC')
WHERE budget_id = ANY($1::text[])
"""
_ALIGN_RESET_TO_DEFAULT_SQL: Final = """
UPDATE "LiteLLM_BudgetTable" b
SET budget_reset_at = (SELECT d.budget_reset_at FROM "LiteLLM_BudgetTable" d WHERE d.budget_id = $2)
FROM "LiteLLM_TeamMembership" m
WHERE m.team_id = $1 AND m.user_id = ANY($3::text[]) AND m.budget_id = b.budget_id AND b.budget_id <> $2
"""
_SHARED_BUDGET_IDS_SQL: Final = """
SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE budget_id = ANY($1::text[])
GROUP BY budget_id HAVING COUNT(*) > 1
"""
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
_CANDIDATES_SQL: Final = (
    "SELECT m.user_id, b.budget_id, b.soft_budget, b.max_parallel_requests, b.tpm_limit, b.rpm_limit, "
    "b.model_max_budget, b.budget_duration, b.allowed_models, b.temp_budget_increase, b.temp_budget_expiry "
    'FROM "LiteLLM_TeamMembership" m JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id WHERE '
    + _ON_CHOSEN_SIDE_OF_DEFAULT
    + " AND (b.budget_duration = ANY($5::text[]) OR ($6::boolean AND b.budget_duration IS NULL))"
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


class _BudgetLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    model_max_budget: JsonValue = None
    budget_duration: str | None = None
    allowed_models: tuple[str, ...] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None

    @field_validator("model_max_budget", mode="before")
    @classmethod
    def _decode_json_column(cls, value: object) -> object:
        return _JSON_VALUE.validate_json(value) if isinstance(value, str) else value

    def has_own_limits_beyond(self, default: "_BudgetLimits") -> bool:
        own: Final = _JSON_OBJECT.validate_python(self.model_dump(mode="json"))
        inherited: Final = _JSON_OBJECT.validate_python(default.model_dump(mode="json"))
        return has_meaningful_budget_limit(
            {field: value for field, value in own.items() if field in _ALWAYS_OWN_LIMITS or value != inherited[field]}
        )


class _MemberBudgetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    budget_id: str


@dataclass(frozen=True, slots=True)
class _CustomMemberBudget:
    ref: _MemberBudgetRef
    keeps_own_limits: bool


class _BudgetIdRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_id: str


_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SCHEDULE_COUNTS: Final = TypeAdapter(tuple[_ScheduleCount, ...])
_MEMBER_BUDGET_REFS: Final = TypeAdapter(tuple[_MemberBudgetRef, ...])
_MEMBER_BUDGET_LIMITS: Final = TypeAdapter(tuple[_BudgetLimits, ...])
_BUDGET_IDS: Final = TypeAdapter(tuple[_BudgetIdRow, ...])


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
    changed_by: UserAPIKeyAuth,
) -> MovedMemberBudgets | TooManyMemberBudgetsToMove:
    """Members compare on their permanent ``max_budget``, and only those resetting on the default's schedule move.
    A member whose other limits all match the default is linked to the default row. One holding a limit of its own, or
    any model list, keeps its row with ``max_budget`` cleared and its reset moved onto the default's."""
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
    rows: Final = await tx.query_raw(
        _CANDIDATES_SQL,
        team_id,
        default_budget_id,
        backfill,
        default.max_budget,
        [count.duration for count in matching if count.duration is not None],
        any(count.duration is None for count in matching),
    )
    default_limits: Final = _BudgetLimits.model_validate(default.model_dump())
    candidates: Final = tuple(
        _CustomMemberBudget(ref=ref, keeps_own_limits=limits.has_own_limits_beyond(default_limits))
        for ref, limits in zip(
            _MEMBER_BUDGET_REFS.validate_python(rows), _MEMBER_BUDGET_LIMITS.validate_python(rows), strict=True
        )
    )
    await _move_to_default(tx, candidates, team_id=team_id, default_budget_id=default_budget_id, changed_by=changed_by)
    return MovedMemberBudgets(
        user_ids=tuple(candidate.ref.user_id for candidate in candidates), skipped_other_reset_period=skipped
    )


async def _move_to_default(
    tx: "Prisma",
    candidates: Sequence[_CustomMemberBudget],
    *,
    team_id: str,
    default_budget_id: str,
    changed_by: UserAPIKeyAuth,
) -> None:
    keeping: Final = tuple(candidate.ref for candidate in candidates if candidate.keeps_own_limits)
    relinking: Final = tuple(candidate.ref.user_id for candidate in candidates if not candidate.keeps_own_limits)
    shared: Final = frozenset(
        row.budget_id
        for row in _BUDGET_IDS.validate_python(
            await tx.query_raw(_SHARED_BUDGET_IDS_SQL, list({ref.budget_id for ref in keeping}))
        )
    )
    await tx.execute_raw(_RELINK_TO_DEFAULT_SQL, team_id, default_budget_id, list(relinking))
    await tx.execute_raw(
        _CLEAR_MAX_BUDGET_SQL,
        list({ref.budget_id for ref in keeping if ref.budget_id not in shared}),
        changed_by.user_id or "",
    )
    for ref in keeping:
        if ref.budget_id in shared:
            await upsert_budget_and_membership(
                tx,
                team_id=team_id,
                user_id=ref.user_id,
                existing_budget_id=ref.budget_id,
                user_api_key_dict=changed_by,
                budget_patch=MappingProxyType({"max_budget": None}),
                team_default_budget_id=default_budget_id,
                shared_budget_ids=shared,
            )
    await tx.execute_raw(_ALIGN_RESET_TO_DEFAULT_SQL, team_id, default_budget_id, [ref.user_id for ref in keeping])


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
