"""`GET /management/v1/budgets`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

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
    OrderBy,
    Scope,
    ScopeAll,
    ScopeDenied,
    SortKey,
    Where,
    handle_list,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    ListResponse,
    ProblemDetail,
)

router = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


class BudgetRow(BaseModel):
    """The `LiteLLM_BudgetTable` columns this list serves.

    Validating the untyped Prisma row through here is what makes `tpm_limit` /
    `rpm_limit` ints: they are `BigInt?` in the schema, which the query engine can
    hand back as a decimal string.
    """

    model_config = ConfigDict(from_attributes=True)

    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


_BUDGET_ROWS = TypeAdapter(tuple[BudgetRow, ...])


@dataclass(frozen=True, slots=True)
class PrismaBudgetListExecutor:
    """The `ListExecutor` half of the budgets list: everything Prisma-shaped lives here."""

    prisma_client: PrismaClient

    async def count(self, where: Where) -> int:
        return int(await self.prisma_client.db.litellm_budgettable.count(where=dict(where)))

    async def find_many(self, where: Where, order: OrderBy, skip: int, take: int) -> Sequence[BudgetRow]:
        rows = await self.prisma_client.db.litellm_budgettable.find_many(
            where=dict(where), order=list(order), skip=skip, take=take
        )
        return _BUDGET_ROWS.validate_python(rows)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize(row: BudgetRow) -> Mapping[str, JsonValue]:
    return {
        "budget_id": row.budget_id,
        "max_budget": row.max_budget,
        "soft_budget": row.soft_budget,
        "tpm_limit": row.tpm_limit,
        "rpm_limit": row.rpm_limit,
        "budget_duration": row.budget_duration,
        "budget_reset_at": _iso(row.budget_reset_at),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _scope(caller: UserAPIKeyAuth) -> Scope:
    if user_api_key_has_admin_view(caller):
        return ScopeAll()
    return ScopeDenied(
        detail="Only proxy admins can list budgets, your role={}".format(caller.user_role),
    )


# budget_duration is deliberately absent from `sortable`: the column holds strings
# like "7d" and "30d", so a lexicographic ORDER BY puts "30d" ahead of "7d".
BUDGETS_LIST_SPEC: ListSpec[BudgetRow] = ListSpec(
    resource="budgets",
    sortable=frozenset({"budget_id", "max_budget", "tpm_limit", "rpm_limit", "created_at"}),
    searchable=frozenset({"budget_id"}),
    filters={
        "budget_duration": FilterSpec(type="string", ops=frozenset({"in", "is_null"})),
        "max_budget": FilterSpec(type="number", ops=frozenset({"gte", "lte", "is_null"})),
        "created_at": FilterSpec(type="datetime", ops=frozenset({"gte", "lte"})),
    },
    default_sort=(SortKey(field="created_at", descending=True), SortKey(field="budget_id", descending=False)),
    default_page_size=50,
    max_page_size=100,
    scope=_scope,
    serialize=_serialize,
    tiebreaker="budget_id",
)


@router.get(
    "/budgets",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListResponse,
)
async def list_budgets(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ListResponse:
    """
    The budgets defined on this proxy, paged, sortable and filterable, for the
    Budgets page.

    Readable by a proxy admin or an admin viewer; anyone else is refused 403. The
    older `/budget/list` answers with the whole table as a bare array and has no
    way to page, sort or filter it.

    `sort` takes a comma-separated list of `budget_id`, `max_budget`, `tpm_limit`,
    `rpm_limit` or `created_at`, each optionally prefixed with `-` for descending,
    and defaults to `-created_at,budget_id`. `q` is a case-insensitive substring
    match on `budget_id`. `page_size` defaults to 50 and is capped at 100.
    Filters are `filter[budget_duration][in|is_null]`,
    `filter[max_budget][gte|lte|is_null]` and `filter[created_at][gte|lte]`.

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
            request=request,
            spec=BUDGETS_LIST_SPEC,
            executor=PrismaBudgetListExecutor(prisma_client=prisma_client),
            caller=user_api_key_dict,
        )

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.budgets.list_budgets(): Exception occured - {}".format(
                str(e)
            )
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list budgets.",
            )
        )
