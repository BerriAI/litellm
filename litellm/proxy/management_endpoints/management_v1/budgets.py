"""`GET /management/v1/budgets`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    UserAPIKeyAuth,
    user_api_key_has_admin_view,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
)
from litellm.proxy.management_endpoints.management_v1.list_framework import (
    FilterSpec,
    ListSpec,
    Predicate,
    QueryPlan,
    Scope,
    ScopeAll,
    ScopeDenied,
    SortKey,
    handle_list,
    order_by_sql,
    where_sql,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    ListResponse,
    ProblemDetail,
)

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)

BUDGET_TABLE: Final = '"LiteLLM_BudgetTable"'


class BudgetListItem(BaseModel):
    """One budget as the Budgets page reads it, and as it comes back off the table.

    Validating the raw row through here is what makes `tpm_limit` / `rpm_limit`
    numbers: they are `BigInt?` in the schema, which the query engine hands back as
    decimal strings, and a quoted "60000" breaks arithmetic in the dashboard.
    """

    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class _RowCount(BaseModel):
    count: int


_BUDGET_ROWS: Final = TypeAdapter(tuple[BudgetListItem, ...])
_ROW_COUNTS: Final = TypeAdapter(tuple[_RowCount, ...])

SELECTED_COLUMNS: Final = ", ".join(f'"{name}"' for name in BudgetListItem.model_fields)


@dataclass(frozen=True, slots=True)
class PrismaBudgetListExecutor:
    """The database half of the budgets list. Every caller-supplied value is bound to a
    placeholder by `where_sql`; only the spec's own column names reach the SQL text."""

    prisma_client: PrismaClient

    async def count(self, where: tuple[Predicate, ...]) -> int:
        clauses, params = where_sql(where)
        sql: Final = f"SELECT COUNT(*) AS count FROM {BUDGET_TABLE}" + (f" WHERE {clauses}" if clauses else "")
        rows: Final = await self.prisma_client.db.query_raw(sql, *params)
        counted: Final = _ROW_COUNTS.validate_python(rows)
        return counted[0].count if counted else 0

    async def find_many(self, plan: QueryPlan) -> Sequence[BudgetListItem]:
        clauses, params = where_sql(plan.where)
        sql: Final = (
            f"SELECT {SELECTED_COLUMNS} FROM {BUDGET_TABLE}"
            + (f" WHERE {clauses}" if clauses else "")
            + f" ORDER BY {order_by_sql(plan.order)}"
            + f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        )
        rows: Final = await self.prisma_client.db.query_raw(sql, *params, plan.take, plan.skip)
        return _BUDGET_ROWS.validate_python(rows)


def _serialize(row: BudgetListItem) -> BudgetListItem:
    """The row shape is the wire shape: the query selects exactly the columns served."""
    return row


def _scope(caller: UserAPIKeyAuth) -> Scope:
    if user_api_key_has_admin_view(caller):
        return ScopeAll()
    return ScopeDenied(reason=f"Only proxy admins can list budgets, your role={caller.user_role}")


# budget_duration is deliberately absent from `sortable`: the column holds strings
# like "7d" and "30d", so a lexicographic ORDER BY puts "30d" ahead of "7d".
BUDGET_FILTERS: Final[Mapping[str, FilterSpec]] = MappingProxyType(
    {  # mutable-ok: an immutable mapping has no literal form; MappingProxyType freezes this one and it never escapes
        "budget_duration": FilterSpec(type=str, ops=frozenset(("in", "is_null"))),
        "max_budget": FilterSpec(type=float, ops=frozenset(("gte", "lte", "is_null"))),
        "created_at": FilterSpec(type=datetime, ops=frozenset(("gte", "lte"))),
    }
)

BUDGETS_LIST_SPEC: Final[ListSpec[BudgetListItem, BudgetListItem]] = ListSpec(
    resource="budgets",
    sortable=frozenset(("budget_id", "max_budget", "tpm_limit", "rpm_limit", "created_at")),
    searchable=frozenset(("budget_id",)),
    filters=BUDGET_FILTERS,
    default_sort=(SortKey(field="created_at", descending=True),),
    default_page_size=50,
    max_page_size=100,
    scope=_scope,
    serialize=_serialize,
    tiebreaker="budget_id",
)


@router.get(
    "/budgets",
    tags=("budget management",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[BudgetListItem],
)
async def list_budgets(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ListResponse[BudgetListItem]:
    """
    The budgets defined on this proxy, paged, sortable and filterable, for the
    Budgets page.

    Readable by a proxy admin or an admin viewer; anyone else is refused 403. The
    older `/budget/list` answers with the whole table as a bare array and has no
    way to page, sort or filter it.

    `sort` takes a comma-separated list of `budget_id`, `max_budget`, `tpm_limit`,
    `rpm_limit` or `created_at`, each optionally prefixed with `-` for descending,
    and defaults to `-created_at`. `budget_id` is appended to every sort as the
    tiebreaker. `q` is a case-insensitive substring match on `budget_id`.
    `page_size` defaults to 50 and is capped at 100. Filters are
    `filter[budget_duration][in|is_null]`, `filter[max_budget][gte|lte|is_null]`
    and `filter[created_at][gte|lte]`.

    Example curl:
    ```
    curl --location --globoff 'http://0.0.0.0:4000/management/v1/budgets?sort=-max_budget&filter[budget_duration][in]=7d,30d&page_size=25' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        return await handle_list(
            spec=BUDGETS_LIST_SPEC,
            executor=PrismaBudgetListExecutor(prisma_client=prisma_client),
            request=request,
            caller=user_api_key_dict,
        )

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.budgets.list_budgets(): Exception occured - %s", e
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list budgets.",
            )
        )
