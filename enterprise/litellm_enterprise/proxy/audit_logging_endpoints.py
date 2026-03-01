"""
AUDIT LOGGING

All /audit logging endpoints. Attempting to write these as CRUD endpoints.

GET - /audit/{id} - Get audit log by id
GET - /audit - Get all audit logs
"""

from typing import Any, Dict, List, Optional

#### AUDIT LOGGING ####
from fastapi import APIRouter, Depends, HTTPException, Query
from litellm_enterprise.types.proxy.audit_logging_endpoints import (
    AuditLogResponse,
    PaginatedAuditLogResponse,
)

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _build_json_field_conditions(
    field: str, json_key: str, value: str
) -> List[Dict[str, Any]]:
    """
    Build OR conditions to match a value inside a JSON column at the given key.

    Uses Prisma's JSON path filtering (PostgreSQL only). Returns a list of
    two conditions — one for `before_value` and one for `updated_values` — to
    be merged into the caller's top-level OR list.
    """
    return [
        {field: {"path": [json_key], "string_contains": value}},
    ]


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
    changed_by: Optional[str] = Query(
        None, description="Filter by user or system that performed the action"
    ),
    changed_by_api_key: Optional[str] = Query(
        None, description="Filter by API key hash that performed the action"
    ),
    action: Optional[str] = Query(
        None, description="Filter by action type (create, update, delete)"
    ),
    table_name: Optional[str] = Query(
        None, description="Filter by table name that was modified"
    ),
    object_id: Optional[str] = Query(
        None, description="Filter by ID of the object that was modified"
    ),
    start_date: Optional[str] = Query(None, description="Filter logs after this date"),
    end_date: Optional[str] = Query(None, description="Filter logs before this date"),
    object_team_id: Optional[str] = Query(
        None,
        description="Filter by team_id present in before_value or updated_values JSON (PostgreSQL only)",
    ),
    object_key_hash: Optional[str] = Query(
        None,
        description="Filter by token (key hash) present in before_value or updated_values JSON (PostgreSQL only)",
    ),
    # Sorting parameters
    sort_by: Optional[str] = Query(
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
    where_conditions: Dict[str, Any] = {}
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
        date_filter: Dict[str, Any] = {}
        if start_date:
            date_filter["gte"] = start_date
        if end_date:
            date_filter["lte"] = end_date
        where_conditions["updated_at"] = date_filter

    # JSON field filters (PostgreSQL only) — search inside before_value and
    # updated_values for a matching key/value pair.
    if object_team_id:
        where_conditions["OR"] = [
            *_build_json_field_conditions("before_value", "team_id", object_team_id),
            *_build_json_field_conditions("updated_values", "team_id", object_team_id),
        ]
    if object_key_hash:
        existing_or: List[Dict[str, Any]] = where_conditions.get("OR", [])
        where_conditions["OR"] = [
            *existing_or,
            *_build_json_field_conditions("before_value", "token", object_key_hash),
            *_build_json_field_conditions("updated_values", "token", object_key_hash),
        ]

    # Build sort conditions
    order_by: Dict[str, Any] = {}
    if sort_by and isinstance(sort_by, str):
        order_by[sort_by] = sort_order
    else:
        order_by["updated_at"] = sort_order  # Default sort by updated_at

    # Get paginated results
    audit_logs = await prisma_client.db.litellm_auditlog.find_many(
        where=where_conditions,
        order=order_by,
        skip=(page - 1) * page_size,
        take=page_size,
    )

    # Get total count for pagination
    total_count = await prisma_client.db.litellm_auditlog.count(where=where_conditions)
    total_pages = -(-total_count // page_size)  # Ceiling division

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

    # Get the audit log by ID
    audit_log = await prisma_client.db.litellm_auditlog.find_unique(where={"id": id})

    if audit_log is None:
        raise HTTPException(
            status_code=404, detail={"message": f"Audit log with ID {id} not found"}
        )

    # Convert to response model
    return AuditLogResponse(**audit_log.model_dump())
