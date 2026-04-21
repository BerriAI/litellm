"""
AUDIT LOG ENDPOINTS

GET /audit/logs  - Query audit logs (admin-only, paginated, filterable)
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _get_read_prisma_client():
    """Use read replica for audit log queries, falling back to primary."""
    from litellm.proxy.proxy_server import prisma_client, prisma_read_client

    client = prisma_read_client if prisma_read_client is not None else prisma_client
    if client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return client


@router.get(
    "/audit/logs",
    tags=["Audit Logs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_audit_logs(
    action: Optional[str] = Query(default=None, description="Filter by action: created, updated, deleted"),
    table_name: Optional[str] = Query(default=None, description="Filter by table: LiteLLM_UserTable, LiteLLM_VerificationToken, LiteLLM_TeamTable, LiteLLM_ModelTable"),
    object_id: Optional[str] = Query(default=None, description="Filter by object ID"),
    changed_by: Optional[str] = Query(default=None, description="Filter by user who made the change"),
    start_date: Optional[str] = Query(default=None, description="Start date filter (ISO 8601, e.g. 2024-01-01T00:00:00Z)"),
    end_date: Optional[str] = Query(default=None, description="End date filter (ISO 8601, e.g. 2024-12-31T23:59:59Z)"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=50, ge=1, le=500, description="Number of records per page"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Query audit logs. Admin-only endpoint.

    Supports filtering by action, table, object_id, changed_by, and date range.
    Returns paginated results.
    """
    from litellm.proxy._types import LitellmUserRoles

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(status_code=403, detail={"error": "Only admins can view audit logs."})

    prisma_client = _get_read_prisma_client()

    where: dict = {}

    if action:
        where["action"] = action
    if table_name:
        where["table_name"] = table_name
    if object_id:
        where["object_id"] = object_id
    if changed_by:
        where["changed_by"] = changed_by

    date_filter: dict = {}
    if start_date:
        try:
            date_filter["gte"] = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": f"Invalid start_date format: {start_date}"})
    if end_date:
        try:
            date_filter["lte"] = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": f"Invalid end_date format: {end_date}"})
    if date_filter:
        where["updated_at"] = date_filter

    skip = (page - 1) * page_size

    try:
        total = await prisma_client.db.litellm_auditlog.count(where=where)  # type: ignore
        logs = await prisma_client.db.litellm_auditlog.find_many(
            where=where,  # type: ignore
            skip=skip,
            take=page_size,
            order={"updated_at": "desc"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Failed to fetch audit logs: {str(e)}"})

    return {
        "data": [log.model_dump() for log in logs],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }
