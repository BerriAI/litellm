"""
AUDIT LOGGING

All /audit logging endpoints. Attempting to write these as CRUD endpoints.

GET - /audit/{id} - Get audit log by id
GET - /audit - Get all audit logs
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, Protocol

#### AUDIT LOGGING ####
from fastapi import APIRouter, Depends, HTTPException, Query
from litellm_enterprise.types.proxy.audit_logging_endpoints import (
    AuditLogResponse,
    PaginatedAuditLogResponse,
)
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class _AuditLogFields(TypedDict):
    """Columns of the `LiteLLM_AuditLog` table, as returned by `model_dump()`."""

    id: ReadOnly[str]
    updated_at: ReadOnly[datetime]
    changed_by: ReadOnly[str]
    changed_by_api_key: ReadOnly[str]
    action: ReadOnly[str]
    table_name: ReadOnly[str]
    object_id: ReadOnly[str]
    before_value: ReadOnly[dict[str, object] | None]
    updated_values: ReadOnly[dict[str, object] | None]


class _AuditLogRecord(Protocol):
    """Row of the `LiteLLM_AuditLog` table as materialised by the Prisma client."""

    def model_dump(self) -> _AuditLogFields: ...


class _AuditLogTable(Protocol):
    """The `litellm_auditlog` accessor of the Prisma client."""

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        order: Mapping[str, str],
        skip: int,
        take: int,
    ) -> Sequence[_AuditLogRecord]: ...

    async def count(self, *, where: Mapping[str, object]) -> int: ...

    async def find_unique(self, *, where: Mapping[str, str]) -> _AuditLogRecord | None: ...


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

    # Build filter conditions
    where_conditions: Final[dict[str, object]] = {}
    if changed_by:
        where_conditions["changed_by"] = changed_by
    if changed_by_api_key:
        where_conditions["changed_by_api_key"] = changed_by_api_key
    if action:
        where_conditions["action"] = action
    if table_name:
        where_conditions["table_name"] = table_name
    if object_id:
        where_conditions["object_id"] = object_id
    if start_date or end_date:
        date_filter: Final[Mapping[str, str]] = {
            bound: bound_value for bound, bound_value in (("gte", start_date), ("lte", end_date)) if bound_value
        }
        where_conditions["updated_at"] = date_filter

    # JSON field filters (PostgreSQL only) — each filter is AND'd with the
    # others, but checks both before_value and updated_values internally (OR).
    if object_team_id or object_key_hash:
        where_conditions["AND"] = [
            _build_json_field_or_condition(json_key, json_value)
            for json_key, json_value in (
                ("team_id", object_team_id),
                ("token", object_key_hash),
            )
            if json_value
        ]

    # Build sort conditions
    sort_column: Final[str] = sort_by if sort_by and isinstance(sort_by, str) else "updated_at"
    order_by: Final[Mapping[str, str]] = {sort_column: sort_order}

    audit_log_table: Final[_AuditLogTable] = prisma_client.db.litellm_auditlog

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
            AuditLogResponse(**audit_log.model_dump()) for audit_log in audit_logs
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

    audit_log_table: Final[_AuditLogTable] = prisma_client.db.litellm_auditlog

    # Get the audit log by ID
    audit_log: Final = await audit_log_table.find_unique(where={"id": id})

    if audit_log is None:
        raise HTTPException(
            status_code=404, detail={"message": f"Audit log with ID {id} not found"}
        )

    # Convert to response model
    return AuditLogResponse(**audit_log.model_dump())
