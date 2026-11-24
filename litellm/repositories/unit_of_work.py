"""
Units of work over a single Prisma batch.

Each context manager here opens one ``db.batch_()`` and binds a typed write
repository per table to it, so every update queued through the yielded object
lands in the same transaction. The batch commits when the block exits cleanly
and is abandoned, writing nothing, when the block raises.

``spend_reset_unit_of_work`` covers the per-row key/user/team resets;
``budget_cascade_unit_of_work`` covers a budget tier's reset, where the
dependent spend and the tier's next window have to move together.

Each write repository queues narrow ``{spend}`` / ``{budget_reset_at}`` updates
instead of full-model writes, which trip ``prisma.errors.DataError`` on rows
carrying fields the update input type rejects (see #27730).
"""

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, ulp
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

from litellm.repositories.prisma_protocols import BatchTable, PrismaBatch


def _spend_reset_data(budget_reset_at: datetime | None, spend_decrement: float) -> Mapping[str, object]:
    spend: Final[object] = {"decrement": spend_decrement}
    return {"spend": spend, "budget_reset_at": budget_reset_at}


_SPEND_RESET_SET: Final = (
    " SET spend = CASE WHEN ABS(spend - $3::text::double precision) <= $5::text::double precision"
    " THEN 0 ELSE GREATEST(spend - $3::text::double precision, 0) END, budget_reset_at = $2::timestamp,"
    " updated_at = (NOW() AT TIME ZONE 'UTC')"
)
_SPEND_RESET_WHERE: Final = " AND budget_reset_at IS NOT DISTINCT FROM $4::timestamp"
_USER_TEMP_BUDGET_WHERE: Final = (
    " AND (temp_budget_increase IS NOT DISTINCT FROM ${amount}::text::double precision"
    " OR ABS(temp_budget_increase::text::numeric - ${amount}::text::numeric)"
    " <= ABS(${amount}::text::numeric) * 0.000000000000001)"
    " AND temp_budget_expiry IS NOT DISTINCT FROM ${expiry}::timestamp"
)
# Evaluate expiry after the CTE acquires the lock, even when the lock holder leaves the row unchanged.
_USER_TEMP_BUDGET_ACTIVE_WHERE: Final = (
    ' AND EXISTS (WITH locked_user AS MATERIALIZED (SELECT temp_budget_expiry FROM "LiteLLM_UserTable"'
    " WHERE user_id = $1 FOR UPDATE)"
    " SELECT 1 FROM locked_user WHERE temp_budget_expiry > (clock_timestamp() AT TIME ZONE 'UTC'))"
)
_KEY_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_VerificationToken"' + _SPEND_RESET_SET + " WHERE token = $1" + _SPEND_RESET_WHERE
)
_USER_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_UserTable"' + _SPEND_RESET_SET + " WHERE user_id = $1" + _SPEND_RESET_WHERE
)
_USER_CREDIT_RESET_SQL: Final = (
    'UPDATE "LiteLLM_UserTable"'
    + _SPEND_RESET_SET
    + ", rollover_credit = CASE WHEN budget_duration IS NOT NULL AND max_budget > 0"
    " AND rollover_max_budget > max_budget AND rollover_max_budget < 'Infinity'::double precision THEN"
    " LEAST(rollover_max_budget - max_budget, GREATEST(0, max_budget"
    " + LEAST(rollover_max_budget - max_budget, GREATEST(0, COALESCE(rollover_credit, 0))) - $6::text::double precision))"
    " ELSE NULL END WHERE user_id = $1" + _SPEND_RESET_WHERE
)
_MEMBER_CREDIT_WINDOW_RESET_SQL: Final = """
WITH budget AS (
    UPDATE "LiteLLM_BudgetTable"
    SET budget_reset_at = $3::timestamp, budget_reset_count = budget_reset_count + 1,
        updated_at = (NOW() AT TIME ZONE 'UTC')
    WHERE budget_id = $1 AND budget_reset_at IS NOT DISTINCT FROM $4::timestamp AND budget_duration = $5::text
    RETURNING budget_id, max_budget,
        CASE WHEN max_budget > 0 AND rollover_max_budget > max_budget
            AND rollover_max_budget < 'Infinity'::double precision
            THEN rollover_max_budget - max_budget END AS credit_room
)
UPDATE "LiteLLM_TeamMembership" AS member
SET rollover_credit = CASE WHEN credit_room IS NOT NULL THEN
        LEAST(credit_room, GREATEST(0, budget.max_budget
            + LEAST(credit_room, GREATEST(0, COALESCE(member.rollover_credit, 0))) - member.spend)) END,
    spend = CASE WHEN $2::boolean AND ABS(budget.max_budget) < 'Infinity'::double precision THEN
        GREATEST(0, member.spend - budget.max_budget
            - LEAST(COALESCE(credit_room, 0), GREATEST(0, COALESCE(member.rollover_credit, 0))))
        ELSE 0 END
FROM budget
WHERE member.budget_id = budget.budget_id
"""
_TEAM_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_TeamTable"' + _SPEND_RESET_SET + " WHERE team_id = $1" + _SPEND_RESET_WHERE
)


def _reset_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    reset_at: Final = (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).astimezone(timezone.utc)
    return reset_at.replace(tzinfo=None).isoformat(timespec="microseconds")


def _queue_guarded_spend_reset(
    batch: PrismaBatch,
    query: str,
    identity: str,
    budget_reset_at: datetime | None,
    spend_decrement: float,
    expected_reset_at: datetime | None,
    spend_after_reset: float,
    *where_values: str | None,
) -> bool:
    if budget_reset_at is None or (spend_decrement == 0 and not where_values) or not isfinite(spend_decrement):
        return False
    # Prisma rounds Float reads, so exact subtraction can still leave a few ULPs.
    tolerance: Final = 4 * ulp(spend_decrement) if spend_after_reset == 0.0 else 0.0
    batch.execute_raw(
        query,
        identity,
        _reset_timestamp(budget_reset_at),
        repr(spend_decrement),
        _reset_timestamp(expected_reset_at),
        str(tolerance),
        *where_values,
    )
    return True


@dataclass(frozen=True, slots=True)
class KeySpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_spend_reset(
        self,
        token: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
    ) -> None:
        if _queue_guarded_spend_reset(
            self.batch,
            _KEY_SPEND_RESET_SQL,
            token,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
        ):
            return
        self.table.update_many(
            where={"token": token, "budget_reset_at": expected_reset_at},
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@dataclass(frozen=True, slots=True)
class UserSpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_credit_reset(
        self,
        user_id: str,
        budget_reset_at: datetime,
        expected_reset_at: datetime | None,
        closed_spend: float,
        spend_decrement: float,
        spend_after_reset: float,
        expected_temp_budget: tuple[float | None, datetime | None] | None = None,
        require_active_temp_budget: bool = False,
    ) -> None:
        guard: Final = self._temp_budget_guard(expected_temp_budget)
        self.batch.execute_raw(
            _USER_CREDIT_RESET_SQL
            + (_USER_TEMP_BUDGET_WHERE.format(amount=7, expiry=8) if guard else "")
            + (_USER_TEMP_BUDGET_ACTIVE_WHERE if require_active_temp_budget else ""),
            user_id,
            _reset_timestamp(budget_reset_at),
            repr(spend_decrement),
            _reset_timestamp(expected_reset_at),
            str(4 * ulp(spend_decrement) if spend_after_reset == 0.0 else 0.0),
            repr(closed_spend),
            *guard,
        )

    def queue_spend_reset(
        self,
        user_id: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
        expected_temp_budget: tuple[float | None, datetime | None] | None = None,
        require_active_temp_budget: bool = False,
    ) -> None:
        guard: Final = self._temp_budget_guard(expected_temp_budget)
        if _queue_guarded_spend_reset(
            self.batch,
            _USER_SPEND_RESET_SQL
            + (_USER_TEMP_BUDGET_WHERE.format(amount=6, expiry=7) if guard else "")
            + (_USER_TEMP_BUDGET_ACTIVE_WHERE if require_active_temp_budget else ""),
            user_id,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
            *guard,
        ):
            return
        self.table.update_many(
            where={
                "user_id": user_id,
                "budget_reset_at": expected_reset_at,
                **(
                    {"temp_budget_increase": expected_temp_budget[0], "temp_budget_expiry": expected_temp_budget[1]}
                    if expected_temp_budget is not None
                    else {}
                ),
            },
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )

    @staticmethod
    def _temp_budget_guard(expected: tuple[float | None, datetime | None] | None) -> tuple[str | None, ...]:
        if expected is None:
            return ()
        amount, expiry = expected
        return (repr(amount) if amount is not None else None, _reset_timestamp(expiry))


@dataclass(frozen=True, slots=True)
class TeamSpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_spend_reset(
        self,
        team_id: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
    ) -> None:
        if _queue_guarded_spend_reset(
            self.batch,
            _TEAM_SPEND_RESET_SQL,
            team_id,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
        ):
            return
        self.table.update_many(
            where={"team_id": team_id, "budget_reset_at": expected_reset_at},
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@runtime_checkable
class SpendResetWrites(Protocol):
    def queue_spend_zero(self, where: Mapping[str, object]) -> None: ...

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None: ...


@dataclass(frozen=True, slots=True)
class LinkedSpendResetWrites:
    table: BatchTable

    def queue_spend_zero(self, where: Mapping[str, object]) -> None:
        self.table.update_many(where=where, data={"spend": 0})

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None:
        """``decrement`` rather than a read-then-set, so spend written between the
        cascade's read and its commit survives the reset instead of being erased."""
        self.table.update_many(
            where=where,
            data={"spend": {"decrement": amount}},
        )


@dataclass(frozen=True, slots=True)
class WindowedSpendResetWrites:
    """End-user and customer team and member pages commit ahead of the window
    advance, so a failed advance replays them. Each reset stamps the tier window
    it closes and skips rows already stamped with it: replaying the cap
    predicate would zero a carried overage, and replaying the zero would erase
    spend accrued since. Without a window the resets are unguarded."""

    table: BatchTable

    def queue_spend_zero(self, where: Mapping[str, object], window: str | None) -> None:
        self.table.update_many(where=_not_reset_in(window, where), data=_stamped(window, {"spend": 0}))

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float, window: str | None) -> None:
        self.table.update_many(
            where=_not_reset_in(window, where),
            data=_stamped(window, {"spend": {"decrement": amount}}),
        )


def _stamped(window: str | None, data: Mapping[str, object]) -> Mapping[str, object]:
    if window is None:
        return data
    return {**data, "spend_reset_window": window}


def _not_reset_in(window: str | None, where: Mapping[str, object]) -> Mapping[str, object]:
    if window is None:
        return where
    return {
        "AND": [
            {**where},
            {"OR": [{"spend_reset_window": None}, {"spend_reset_window": {"not": window}}]},
        ]
    }


# Membership rows written before total_spend existed (migration 20260421135425, default 0) hold their
# history only in spend. Zeroing spend would erase it, so the floor lands in the same batch, ahead of the reset.
_TEAM_MEMBERSHIP_TOTAL_SPEND_REPAIR_SQL: Final = """
UPDATE "LiteLLM_TeamMembership"
SET total_spend = spend
WHERE budget_id = ANY($1::text[]) AND spend > total_spend
"""


@dataclass(frozen=True, slots=True)
class TeamMembershipSpendResetWrites:
    resets: LinkedSpendResetWrites
    batch: PrismaBatch

    def queue_spend_zero(self, where: Mapping[str, object]) -> None:
        self.resets.queue_spend_zero(where)

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None:
        self.resets.queue_spend_decrement(where, amount)

    def queue_total_spend_repair(self, budget_ids: Sequence[str]) -> None:
        self.batch.execute_raw(_TEAM_MEMBERSHIP_TOTAL_SPEND_REPAIR_SQL, list(budget_ids))


_LOCK_BUDGET_ORGS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_OrganizationTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY organization_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_MEMBERSHIPS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_TeamMembership" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY team_id COLLATE "C", user_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_KEYS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_VerificationToken" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY token COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_PROJECTS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_ProjectTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY project_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_TAGS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_TagTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY tag_name COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_ACCESS_GROUPS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY access_group_name COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_CUSTOMER_ORGS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_CustomerOrganizationTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY customer_org_id COLLATE "C" FOR UPDATE'
)


@dataclass(frozen=True, slots=True)
class BudgetLinkedRowLocks:
    batch: PrismaBatch

    def queue_row_locks(self, budget_ids: Sequence[str]) -> None:
        """Locks every row a cascade may reset, in the shared row lock order, before the filtered resets run."""
        self.batch.execute_raw(_LOCK_BUDGET_ORGS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_MEMBERSHIPS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_KEYS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_PROJECTS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_TAGS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_ACCESS_GROUPS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_CUSTOMER_ORGS_SQL, list(budget_ids))


@dataclass(frozen=True, slots=True)
class BudgetWindowWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_window_advance(self, budget_id: str, budget_reset_at: datetime) -> None:
        """``update_many`` so a tier deleted between the read and the commit is a
        no-op row count instead of a P2025 that aborts the whole chunk. The
        count names the tier's windows for the end-user reset stamps."""
        self.table.update_many(
            where={"budget_id": budget_id},
            data={"budget_reset_at": budget_reset_at, "budget_reset_count": {"increment": 1}},
        )

    def queue_member_credit_window_advance(
        self,
        budget_id: str,
        *,
        carry_overage: bool,
        budget_reset_at: datetime,
        expected_reset_at: datetime | None,
        expected_duration: str,
    ) -> None:
        self.batch.execute_raw(
            _MEMBER_CREDIT_WINDOW_RESET_SQL,
            budget_id,
            carry_overage,
            _reset_timestamp(budget_reset_at),
            _reset_timestamp(expected_reset_at),
            expected_duration,
        )


@dataclass(frozen=True, slots=True)
class SpendResetUnitOfWork:
    keys: KeySpendResetWrites
    users: UserSpendResetWrites
    teams: TeamSpendResetWrites


@dataclass(frozen=True, slots=True)
class BudgetCascadeUnitOfWork:
    """Every write a budget-tier reset performs in its final batch.

    The dependent spend rows and the budget rows' ``budget_reset_at`` advance
    must land together: advancing the window without zeroing the spend it
    gates leaves the dependents pinned at their cap until the next window.
    End users are the exception, paged and committed ahead of this batch by
    ``enduser_spend_reset_unit_of_work`` because one tier can gate millions.
    """

    row_locks: BudgetLinkedRowLocks
    team_memberships: TeamMembershipSpendResetWrites
    keys: LinkedSpendResetWrites
    organizations: LinkedSpendResetWrites
    tags: LinkedSpendResetWrites
    model_access_groups: LinkedSpendResetWrites
    projects: LinkedSpendResetWrites
    customer_organizations: LinkedSpendResetWrites
    budgets: BudgetWindowWrites


@asynccontextmanager
async def spend_reset_unit_of_work(new_batch: Callable[[], PrismaBatch]) -> AsyncGenerator[SpendResetUnitOfWork, None]:
    batch = new_batch()
    yield SpendResetUnitOfWork(
        keys=KeySpendResetWrites(table=batch.litellm_verificationtoken, batch=batch),
        users=UserSpendResetWrites(table=batch.litellm_usertable, batch=batch),
        teams=TeamSpendResetWrites(table=batch.litellm_teamtable, batch=batch),
    )
    await batch.commit()


@asynccontextmanager
async def budget_cascade_unit_of_work(
    new_batch: Callable[[], PrismaBatch],
) -> AsyncGenerator[BudgetCascadeUnitOfWork, None]:
    batch = new_batch()
    yield BudgetCascadeUnitOfWork(
        row_locks=BudgetLinkedRowLocks(batch=batch),
        team_memberships=TeamMembershipSpendResetWrites(
            resets=LinkedSpendResetWrites(table=batch.litellm_teammembership), batch=batch
        ),
        keys=LinkedSpendResetWrites(table=batch.litellm_verificationtoken),
        organizations=LinkedSpendResetWrites(table=batch.litellm_organizationtable),
        tags=LinkedSpendResetWrites(table=batch.litellm_tagtable),
        model_access_groups=LinkedSpendResetWrites(table=batch.litellm_modelaccessgroupbudgettable),
        projects=LinkedSpendResetWrites(table=batch.litellm_projecttable),
        customer_organizations=LinkedSpendResetWrites(table=batch.litellm_customerorganizationtable),
        budgets=BudgetWindowWrites(table=batch.litellm_budgettable, batch=batch),
    )
    await batch.commit()


CustomerSpendTable: TypeAlias = Literal["litellm_customerteamtable", "litellm_customermembertable"]


@asynccontextmanager
async def customer_spend_reset_unit_of_work(
    new_batch: Callable[[], PrismaBatch], table: CustomerSpendTable
) -> AsyncGenerator[WindowedSpendResetWrites, None]:
    """One page of customer team or member spend resets, committed on its own like end users: an aliased request
    can register a member, so the rows a template governs are unbounded."""
    batch: Final = new_batch()
    yield WindowedSpendResetWrites(
        table=batch.litellm_customerteamtable
        if table == "litellm_customerteamtable"
        else batch.litellm_customermembertable
    )
    await batch.commit()


@asynccontextmanager
async def enduser_spend_reset_unit_of_work(
    new_batch: Callable[[], PrismaBatch],
) -> AsyncGenerator[WindowedSpendResetWrites, None]:
    """One page of end-user spend resets, committed on its own so a statement
    never grows with the end-user population."""
    batch: Final = new_batch()
    yield WindowedSpendResetWrites(table=batch.litellm_endusertable)
    await batch.commit()
