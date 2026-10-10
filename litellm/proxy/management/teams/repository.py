"""Team roster reads.

`members_with_roles` on the team row is the roster, as it is for `/team/info` and every access check. A
roster entry can have no `LiteLLM_TeamMembership` row, so membership rows only add spend and limits, and the
user row only fills in the email and alias the roster entry lacks.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.list_api.list_framework import Predicate, QueryPlan, order_by_sql, where_sql


class RawQuery(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class TeamMemberRow(BaseModel):
    user_id: str | None
    user_email: str | None
    user_alias: str | None
    role: Literal["admin", "user"]
    spend: float
    total_spend: float
    budget_id: str | None
    team_default_budget_id: str | None
    max_budget_in_team: float | None
    budget_duration: str | None
    budget_reset_at: datetime | None
    tpm_limit: int | None
    rpm_limit: int | None
    allowed_models: tuple[str, ...]


class _RowCount(BaseModel):
    count: int


_MEMBER_ROWS: Final = TypeAdapter(tuple[TeamMemberRow, ...])
_ROW_COUNTS: Final = TypeAdapter(tuple[_RowCount, ...])

_TEAM_MEMBERS_SQL: Final = """
SELECT
    roster.position,
    roster.member->>'user_id' AS user_id,
    COALESCE(NULLIF(roster.member->>'user_email', ''), u.user_email) AS user_email,
    u.user_alias,
    roster.member->>'role' AS role,
    COALESCE(m.spend, 0) AS spend,
    COALESCE(m.total_spend, 0) AS total_spend,
    m.budget_id,
    team_default.budget_id AS team_default_budget_id,
    b.max_budget AS max_budget_in_team,
    b.budget_duration,
    b.budget_reset_at,
    b.tpm_limit,
    b.rpm_limit,
    COALESCE(b.allowed_models, ARRAY[]::text[]) AS allowed_models
FROM "LiteLLM_TeamTable" t
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(t.members_with_roles) = 'array' THEN t.members_with_roles ELSE '[]'::jsonb END
) WITH ORDINALITY AS roster(member, position)
LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = roster.member->>'user_id'
LEFT JOIN "LiteLLM_TeamMembership" m ON m.team_id = t.team_id AND m.user_id = roster.member->>'user_id'
LEFT JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id
LEFT JOIN "LiteLLM_BudgetTable" team_default ON team_default.budget_id = t.metadata->>'team_member_budget_id'
WHERE t.team_id = $1
"""


@dataclass(frozen=True, slots=True)
class TeamMemberRows:
    """One team's roster. The team is bound here rather than planned, so no query parameter can widen a
    page past that team."""

    db: RawQuery
    team_id: str

    async def count(self, where: tuple[Predicate, ...]) -> int:
        clauses, params = where_sql(where, first_index=2)
        sql: Final = f"SELECT COUNT(*) AS count FROM ({_TEAM_MEMBERS_SQL}) members" + (
            f" WHERE {clauses}" if clauses else ""
        )
        counted: Final = _ROW_COUNTS.validate_python(await self.db.query_raw(sql, self.team_id, *params))
        return counted[0].count if counted else 0

    async def find_many(self, plan: QueryPlan) -> Sequence[TeamMemberRow]:
        clauses, params = where_sql(plan.where, first_index=2)
        sql: Final = (
            f"SELECT * FROM ({_TEAM_MEMBERS_SQL}) members"
            + (f" WHERE {clauses}" if clauses else "")
            + f" ORDER BY {order_by_sql(plan.order)}"
            + f" LIMIT ${len(params) + 2} OFFSET ${len(params) + 3}"
        )
        return _MEMBER_ROWS.validate_python(await self.db.query_raw(sql, self.team_id, *params, plan.take, plan.skip))
