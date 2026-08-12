"""`/management/v1/spend_logs` facets."""

from datetime import datetime, timezone
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Query, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_page_links,
    escape_like,
    reject_unknown_query_params,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    FacetListResponse,
    PageMeta,
    ProblemDetail,
)

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)

# Rows the facet query may read out of LiteLLM_SpendLogs before DISTINCT. Matches
# SPEND_LOGS_PAGINATION_COUNT_CAP, the bound ui_view_spend_logs puts on its count
# query, so both reads of the same table stop at the same depth.
SPEND_LOGS_FACET_SCAN_CAP: Final = 10000


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


async def _end_user_scope_clause(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    next_param_index: int,
) -> tuple[str | None, tuple[Any, ...]]:
    """SQL predicate restricting the facet to spend logs this caller may read.

    Returns ``(None, ())`` for a proxy admin. Mirrors the scoping ``/spend/logs/ui``
    applies, so the dropdown can never offer an end user whose rows the caller
    could not open.
    """
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _get_permitted_team_ids_for_spend_logs,
        _is_admin_view_safe,
    )

    if _is_admin_view_safe(user_api_key_dict=user_api_key_dict):
        return None, ()

    try:
        permitted_team_ids = await _get_permitted_team_ids_for_spend_logs(
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
        )
    except Exception:
        permitted_team_ids = []

    caller_user_id: Final = user_api_key_dict.user_id
    # = ANY(::text[]) rather than an expanded IN list, matching the clause
    # ui_view_spend_logs builds: one parameter whatever the team count.
    templates: Final = (('"user" = ${}',) if caller_user_id is not None else ()) + (
        ("team_id = ANY(${}::text[])",) if permitted_team_ids else ()
    )
    params: Final = ((caller_user_id,) if caller_user_id is not None else ()) + (
        (permitted_team_ids,) if permitted_team_ids else ()
    )
    if not templates:
        return "FALSE", ()
    clauses: Final = tuple(template.format(next_param_index + offset) for offset, template in enumerate(templates))
    return f"({' OR '.join(clauses)})", params


@router.get(
    "/spend_logs/end_users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth), Depends(reject_unknown_query_params)],
    response_model=FacetListResponse,
)
async def list_spend_log_end_users(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_time: Annotated[
        datetime,
        Query(alias="filter[startTime][gte]", description="Window start (UTC when no offset is given)"),
    ],
    end_time: Annotated[
        datetime,
        Query(alias="filter[startTime][lte]", description="Window end (UTC when no offset is given)"),
    ],
    q: Annotated[str | None, Query(description="Case-insensitive partial match on the end user id")] = None,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Page size")] = 50,
) -> FacetListResponse:
    """
    The distinct end users appearing in spend logs over a time window, for the logs
    page filter dropdown.

    Scoped like `/spend/logs/ui`: a proxy admin sees every end user in the window,
    anyone else sees only end users from their own requests or from teams they
    administer (or hold the `/spend/logs` permission on).

    The window is required and the inner scan is capped at SPEND_LOGS_FACET_SCAN_CAP
    rows, so the query cannot degrade into a full-table scan the way
    `/global/all_end_users` does.

    Example curl:
    ```
    curl --location --globoff 'http://0.0.0.0:4000/management/v1/spend_logs/end_users?filter[startTime][gte]=2026-07-23T00:00:00Z&filter[startTime][lte]=2026-07-24T00:00:00Z&page_size=50&q=acme' \
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

        window_params: Final[tuple[Any, ...]] = (_as_utc(start_time), _as_utc(end_time))
        search_params: Final[tuple[Any, ...]] = (f"%{escape_like(q)}%",) if q else ()
        search_clause: Final = (f"end_user ILIKE ${len(window_params) + 1} ESCAPE '\\'",) if q else ()

        scope_clause, scope_params = await _end_user_scope_clause(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            next_param_index=len(window_params) + len(search_params) + 1,
        )

        where_parts: Final = (
            (
                "\"startTime\" >= ($1::timestamptz AT TIME ZONE 'UTC')",
                "\"startTime\" <= ($2::timestamptz AT TIME ZONE 'UTC')",
                "end_user IS NOT NULL",
                "end_user != ''",
            )
            + search_clause
            + ((scope_clause,) if scope_clause is not None else ())
        )

        # The inner LIMIT is the safety bound: it walks the startTime index newest
        # first and stops, so DISTINCT never runs over an unbounded row set.
        # request_id breaks startTime ties so the cut-off row is deterministic and
        # successive OFFSET pages agree on the set they are paging through.
        # page_size + 1: one row beyond the page reveals has_more without a COUNT(*).
        params: Final = (
            window_params
            + search_params
            + scope_params
            + (SPEND_LOGS_FACET_SCAN_CAP, page_size + 1, (page - 1) * page_size)
        )
        scan_idx: Final = len(params) - 2
        facet_sql: Final = (
            f"SELECT DISTINCT end_user FROM ("
            f" SELECT end_user"
            f' FROM "LiteLLM_SpendLogs"'
            f" WHERE {' AND '.join(where_parts)}"
            f' ORDER BY "startTime" DESC, request_id DESC'
            f" LIMIT ${scan_idx}"
            f") recent"
            f" ORDER BY end_user ASC"
            f" LIMIT ${scan_idx + 1} OFFSET ${scan_idx + 2}"
        )
        rows: Final = await prisma_client.db.query_raw(facet_sql, *params)
        end_users: Final[list[str]] = [row["end_user"] for row in rows if row.get("end_user")]
        has_more: Final = len(end_users) > page_size

        return FacetListResponse(
            data=end_users[:page_size],
            meta=PageMeta(page=page, page_size=page_size, has_more=has_more),
            links=build_page_links(request=request, page=page, has_more=has_more),
        )

    except ManagementProblem:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.spend_logs.list_spend_log_end_users(): Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list spend log end users.",
            )
        )
