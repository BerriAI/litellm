from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, TypeAdapter

from litellm.litellm_core_utils.duration_parser import budget_reset_schedule_key

if TYPE_CHECKING:
    from prisma import Prisma


class _BudgetSchedule(BaseModel):
    amount: float
    duration: str | None


class _MemberId(BaseModel):
    user_id: str


_SCHEDULES: Final = TypeAdapter(tuple[_BudgetSchedule, ...])
_MEMBERS: Final = TypeAdapter(tuple[_MemberId, ...])
_DIRECTION: Final = """CASE $3::text
    WHEN 'raise' THEN b.max_budget < $2::double precision
    WHEN 'lower' THEN b.max_budget > $2::double precision
    WHEN 'both' THEN b.max_budget <> $2::double precision
    ELSE FALSE END"""
_RESET: Final = f"""WITH selected AS MATERIALIZED (
    SELECT m.user_id, m.team_id, b.budget_id, (soft_budget IS NOT NULL OR max_parallel_requests IS NOT NULL
        OR tpm_limit IS NOT NULL OR rpm_limit IS NOT NULL OR tpd_limit IS NOT NULL
        OR (model_max_budget IS NOT NULL AND model_max_budget NOT IN ('null'::jsonb, '[]'::jsonb)) OR budget_duration IS NOT NULL
        OR COALESCE(cardinality(allowed_models), 0)>0 OR temp_budget_increase IS NOT NULL OR temp_budget_expiry IS NOT NULL
    ) AS retained
    FROM "LiteLLM_TeamMembership" m
    JOIN "LiteLLM_BudgetTable" b ON b.budget_id=m.budget_id
    WHERE m.team_id=$1 AND b.budget_id<>$6
    AND ({_DIRECTION})
    AND (b.budget_duration=ANY($4::text[])
         OR ($5::boolean AND b.budget_duration IS NULL))
), shared_budgets AS MATERIALIZED (
    SELECT other.budget_id
    FROM "LiteLLM_TeamMembership" other
    WHERE other.budget_id IN (SELECT budget_id FROM selected)
    GROUP BY other.budget_id HAVING COUNT(*)>1
), classified AS MATERIALIZED (
    SELECT s.*,
    (shared_budgets.budget_id IS NOT NULL) AS shared,
    CASE WHEN shared_budgets.budget_id IS NOT NULL THEN gen_random_uuid()::text END AS new_id
    FROM selected s LEFT JOIN shared_budgets ON shared_budgets.budget_id=s.budget_id
), cloned AS (
    INSERT INTO "LiteLLM_BudgetTable" (
        budget_id, soft_budget, max_parallel_requests, tpm_limit, rpm_limit, tpd_limit,
        model_max_budget, budget_duration, budget_reset_at, allowed_models,
        temp_budget_increase, temp_budget_expiry, created_by, updated_by, created_at, updated_at
    ) SELECT new_id, soft_budget, max_parallel_requests, tpm_limit, rpm_limit, tpd_limit,
        model_max_budget, budget_duration, budget_reset_at, allowed_models,
        temp_budget_increase, temp_budget_expiry, $7, $7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    FROM classified c JOIN "LiteLLM_BudgetTable" b ON b.budget_id=c.budget_id
    WHERE c.retained AND c.shared RETURNING budget_id
), cleared AS (
    UPDATE "LiteLLM_BudgetTable" b
    SET max_budget=NULL, updated_by=$7, updated_at=CURRENT_TIMESTAMP
    FROM classified c WHERE b.budget_id=c.budget_id AND c.retained AND NOT c.shared
), linked AS (
    UPDATE "LiteLLM_TeamMembership" m
    SET budget_id=CASE WHEN c.retained THEN cloned.budget_id ELSE NULL END
    FROM classified c LEFT JOIN cloned ON cloned.budget_id=c.new_id
    WHERE m.team_id=$1 AND m.user_id=c.user_id
    AND (NOT c.retained OR cloned.budget_id IS NOT NULL)
) SELECT user_id FROM classified
"""


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
    if mode == "raise":
        return amount < target
    if mode == "lower":
        return amount > target
    return mode == "both" and amount != target


async def reset_member_budget_amounts(
    tx: "Prisma",
    *,
    team_id: str,
    default_budget_id: str,
    target: float,
    target_duration: str | None,
    mode: Literal["keep", "raise", "lower", "both"],
    updated_by: str,
) -> tuple[str, ...]:
    if mode == "keep":
        return ()
    schedules: Final = _SCHEDULES.validate_python(
        await tx.query_raw(
            f"""SELECT MIN(b.max_budget) AS amount, b.budget_duration AS duration
                FROM "LiteLLM_TeamMembership" m
                JOIN "LiteLLM_BudgetTable" b ON b.budget_id=m.budget_id
                WHERE m.team_id=$1 AND b.budget_id<>$4 AND ({_DIRECTION})
                GROUP BY b.budget_duration""",
            team_id,
            target,
            mode,
            default_budget_id,
        )
    )
    matching: Final = tuple(
        schedule.duration
        for schedule in schedules
        if matches_member_budget_update(
            amount=schedule.amount,
            duration=schedule.duration,
            target=target,
            target_duration=target_duration,
            mode=mode,
        )
    )
    if not matching:
        return ()
    durations: Final = [duration for duration in matching if duration is not None]
    changed: Final = _MEMBERS.validate_python(
        await tx.query_raw(_RESET, team_id, target, mode, durations, None in matching, default_budget_id, updated_by)
    )
    return tuple(member.user_id for member in changed)
