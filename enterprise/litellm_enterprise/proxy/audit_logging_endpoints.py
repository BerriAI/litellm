"""
AUDIT LOGGING

All /audit logging endpoints. Attempting to write these as CRUD endpoints.

GET - /audit/{id} - Get audit log by id
GET - /audit - Get all audit logs
"""

from typing import TYPE_CHECKING, Final

#### AUDIT LOGGING ####
from fastapi import APIRouter, Depends, HTTPException, Query
from litellm_enterprise.types.proxy.audit_logging_endpoints import (
    AuditLogResponse,
    PaginatedAuditLogResponse,
)

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import AuditLogRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

router = APIRouter()


def _build_json_field_or_condition(json_key: str, value: str) -> dict[str, object]:
    """
    Build an OR condition that matches a value inside a JSON column at the
    given key, checking both before_value and updated_values.

    Uses Prisma's JSON path filtering (PostgreSQL only).

    Example result (team_id="t1"):
      {"OR": [
          {"before_value":    {"path": ["team_id"], "string_contains": "t1"}},
          {"updated_values":  {"path": ["team_id"], "string_contains": "t1"}},
      ]}
    """
    return {
        "OR": [
            {"before_value": {"path": [json_key], "string_contains": value}},
            {"updated_values": {"path": [json_key], "string_contains": value}},
        ]
    }


@router.get(
    "/audit",
    tags=["Audit Logging"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PaginatedAuditLogResponse,
)
async def get_audit_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    # Filter parameters
    changed_by: str | None = Query(
        None, description="Filter by user or system that performed the action"
    ),
    changed_by_api_key: str | None = Query(
        None, description="Filter by API key hash that performed the action"
    ),
    action: str | None = Query(
        None, description="Filter by action type (create, update, delete)"
    ),
    table_name: str | None = Query(
        None, description="Filter by table name that was modified"
    ),
    object_id: str | None = Query(
        None, description="Filter by ID of the object that was modified"
    ),
    start_date: str | None = Query(None, description="Filter logs after this date"),
    end_date: str | None = Query(None, description="Filter logs before this date"),
    object_team_id: str | None = Query(
        None,
        description="Filter by team_id present in before_value or updated_values JSON (PostgreSQL only)",
    ),
    object_key_hash: str | None = Query(
        None,
        description="Filter by token (key hash) present in before_value or updated_values JSON (PostgreSQL only)",
    ),
    # Sorting parameters
    sort_by: str | None = Query(
        None,
        description="Column to sort by (e.g. 'updated_at', 'action', 'table_name')",
    ),
    sort_order: str = Query("desc", description="Sort order ('asc' or 'desc')"),
):
    """
    Get all audit logs with filtering and pagination.

    Returns a paginated response of audit logs matching the specified filters.

    Note: object_team_id and object_key_hash use Prisma JSON path filtering,
    which requires PostgreSQL.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"message": CommonProxyErrors.db_not_connected_error.value},
        )

    date_filter: Final[dict[str, str]] = {
        **({"gte": start_date} if start_date else {}),
        **({"lte": end_date} if end_date else {}),
    }

    # JSON field filters (PostgreSQL only) — each filter is AND'd with the
    # others, but checks both before_value and updated_values internally (OR).
    json_field_conditions: Final[list[dict[str, object]]] = [
        *([_build_json_field_or_condition("team_id", object_team_id)] if object_team_id else []),
        *([_build_json_field_or_condition("token", object_key_hash)] if object_key_hash else []),
    ]

    # Build filter conditions
    where_conditions: Final[dict[str, object]] = {
        **({"changed_by": changed_by} if changed_by else {}),
        **({"changed_by_api_key": changed_by_api_key} if changed_by_api_key else {}),
        **({"action": action} if action else {}),
        **({"table_name": table_name} if table_name else {}),
        **({"object_id": object_id} if object_id else {}),
        **({"updated_at": date_filter} if start_date or end_date else {}),
        **({"AND": json_field_conditions} if json_field_conditions else {}),
    }

    order_by: Final[dict[str, str]] = (
        {sort_by: sort_order} if sort_by and isinstance(sort_by, str) else {"updated_at": sort_order}
    )

    audit_log_table: Final[TableActions["prisma_models.LiteLLM_AuditLog"]] = AuditLogRepository(prisma_client).table

    # Get paginated results
    audit_logs: Final = await audit_log_table.find_many(
        where=where_conditions,
        order=order_by,
        skip=(page - 1) * page_size,
        take=page_size,
    )

    # Get total count for pagination
    total_count: Final = await audit_log_table.count(where=where_conditions)
    total_pages: Final = -(-total_count // page_size)  # Ceiling division

    # Return paginated response
    return PaginatedAuditLogResponse(
        audit_logs=[
            AuditLogResponse.model_validate(audit_log.model_dump())
            for audit_log in audit_logs
        ]
        if audit_logs
        else [],
        total=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/audit/{id}",
    tags=["Audit Logging"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AuditLogResponse,
    responses={
        404: {"description": "Audit log not found"},
        500: {"description": "Database connection error"},
    },
)
async def get_audit_log_by_id(
    id: str, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
):
    """
    Get detailed information about a specific audit log entry by its ID.

    Args:
        id (str): The unique identifier of the audit log entry

    Returns:
        AuditLogResponse: Detailed information about the audit log entry

    Raises:
        HTTPException: If the audit log is not found or if there's a database connection error
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"message": CommonProxyErrors.db_not_connected_error.value},
        )

    audit_log_table: Final[TableActions["prisma_models.LiteLLM_AuditLog"]] = AuditLogRepository(prisma_client).table

    # Get the audit log by ID
    audit_log: Final = await audit_log_table.find_unique(where={"id": id})

    if audit_log is None:
        raise HTTPException(
            status_code=404, detail={"message": f"Audit log with ID {id} not found"}
        )

    # Convert to response model
    return AuditLogResponse.model_validate(audit_log.model_dump())
