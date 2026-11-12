from collections.abc import Awaitable, Mapping, Sequence
from datetime import date
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.list_api.common import escape_like
from litellm.types.proxy.management_endpoints.customer_usage import CustomerOrganizationUsage


class QueryRows(Protocol):
    def __call__(self, query: str, *args: object) -> Awaitable[Sequence[Mapping[str, object]]]: ...


class CustomerUsagePage(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: tuple[CustomerOrganizationUsage, ...]
    total_count: int


class _Count(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_count: int


_ROWS: Final = TypeAdapter(tuple[CustomerOrganizationUsage, ...])
_COUNTS: Final = TypeAdapter(tuple[_Count, ...])
_SEARCH: Final = r"""LOWER(COALESCE(o.alias, '') || ' ' || o.customer_org_id) LIKE LOWER($1) ESCAPE '\'"""
_COUNT_SQL: Final = f'SELECT COUNT(*) AS total_count FROM "LiteLLM_CustomerOrganizationTable" o WHERE {_SEARCH}'
_USAGE_SQL: Final = rf"""
SELECT o.customer_org_id, o.alias,
       COALESCE(SUM(d.spend), 0) AS spend,
       COALESCE(SUM(d.api_requests), 0) AS requests,
       COALESCE(SUM(d.prompt_tokens + d.completion_tokens), 0) AS tokens
FROM "LiteLLM_CustomerOrganizationTable" o
LEFT JOIN "LiteLLM_CustomerTeamTable" t ON t.parent_org_id = o.customer_org_id
LEFT JOIN "LiteLLM_CustomerMemberTable" m ON m.parent_team_id = t.customer_team_id
LEFT JOIN "LiteLLM_DailyEndUserSpend" d
  ON d.end_user_id = m.customer_member_id AND d.date >= $2 AND d.date <= $3
WHERE {_SEARCH}
GROUP BY o.customer_org_id, o.alias
ORDER BY spend DESC, o.customer_org_id ASC
LIMIT $4 OFFSET $5
"""


class CustomerUsageRepository:
    def __init__(self, query_rows: QueryRows) -> None:
        self._query_rows = query_rows

    async def list_organizations(
        self, start: date, end: date, search: str, page: int, page_size: int
    ) -> CustomerUsagePage:
        pattern: Final = f"%{escape_like(search)}%"
        counts: Final = _COUNTS.validate_python(await self._query_rows(_COUNT_SQL, pattern))
        rows: Final = _ROWS.validate_python(
            await self._query_rows(
                _USAGE_SQL, pattern, start.isoformat(), end.isoformat(), page_size, (page - 1) * page_size
            )
        )
        return CustomerUsagePage(rows=rows, total_count=counts[0].total_count)
