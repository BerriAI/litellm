"""
AUDIT LOGGING

All /audit logging endpoints. Attempting to write these as CRUD endpoints.

GET - /audit/{id} - Get audit log by id
GET - /audit - Get all audit logs
"""

from typing import TYPE_CHECKING, Any, Dict, Final, List, NamedTuple, Optional, Sequence, Tuple

#### AUDIT LOGGING ####
from fastapi import APIRouter, Depends, HTTPException, Query
from litellm_enterprise.types.proxy.audit_logging_endpoints import (
    AuditLogResponse,
    PaginatedAuditLogResponse,
)

from litellm.proxy._types import CommonProxyErrors, LitellmTableNames, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router = APIRouter()

_KEY_TABLE: Final[str] = LitellmTableNames.KEY_TABLE_NAME.value
_TEAM_TABLE: Final[str] = LitellmTableNames.TEAM_TABLE_NAME.value
_USER_TABLE: Final[str] = LitellmTableNames.USER_TABLE_NAME.value
_ORG_TABLE: Final[str] = "LiteLLM_OrganizationTable"
_MODEL_TABLE: Final[str] = LitellmTableNames.PROXY_MODEL_TABLE_NAME.value

_BLOB_ALIAS_KEYS: Final[Dict[str, Tuple[str, ...]]] = {
    _KEY_TABLE: ("key_alias",),
    _TEAM_TABLE: ("team_alias",),
    _USER_TABLE: ("user_alias", "user_email"),
    _ORG_TABLE: ("organization_alias",),
    _MODEL_TABLE: ("model_name",),
}


class _AliasMaps(NamedTuple):
    key_alias_by_token: Dict[str, str]
    team_alias_by_id: Dict[str, str]
    user_alias_by_id: Dict[str, str]
    user_email_by_id: Dict[str, str]
    org_alias_by_id: Dict[str, str]
    model_name_by_id: Dict[str, str]


def _object_ids_for_table(audit_logs: Sequence[AuditLogResponse], table_name: str) -> frozenset:
    return frozenset(log.object_id for log in audit_logs if log.table_name == table_name and log.object_id)


async def _fetch_alias_maps(prisma_client: "PrismaClient", audit_logs: Sequence[AuditLogResponse]) -> _AliasMaps:
    tokens: Final = _object_ids_for_table(audit_logs, _KEY_TABLE) | frozenset(
        log.changed_by_api_key for log in audit_logs if log.changed_by_api_key
    )
    user_ids: Final = _object_ids_for_table(audit_logs, _USER_TABLE) | frozenset(
        log.changed_by for log in audit_logs if log.changed_by
    )
    team_ids: Final = _object_ids_for_table(audit_logs, _TEAM_TABLE)
    org_ids: Final = _object_ids_for_table(audit_logs, _ORG_TABLE)
    model_ids: Final = _object_ids_for_table(audit_logs, _MODEL_TABLE)

    key_rows: Final = (
        await prisma_client.db.litellm_verificationtoken.find_many(where={"token": {"in": list(tokens)}})
        if tokens
        else []
    )
    user_rows: Final = (
        await prisma_client.db.litellm_usertable.find_many(where={"user_id": {"in": list(user_ids)}})
        if user_ids
        else []
    )
    team_rows: Final = (
        await prisma_client.db.litellm_teamtable.find_many(where={"team_id": {"in": list(team_ids)}})
        if team_ids
        else []
    )
    org_rows: Final = (
        await prisma_client.db.litellm_organizationtable.find_many(where={"organization_id": {"in": list(org_ids)}})
        if org_ids
        else []
    )
    model_rows: Final = (
        await prisma_client.db.litellm_proxymodeltable.find_many(where={"model_id": {"in": list(model_ids)}})
        if model_ids
        else []
    )

    return _AliasMaps(
        key_alias_by_token={row.token: row.key_alias for row in key_rows if row.key_alias},
        team_alias_by_id={row.team_id: row.team_alias for row in team_rows if row.team_alias},
        user_alias_by_id={row.user_id: row.user_alias for row in user_rows if row.user_alias},
        user_email_by_id={row.user_id: row.user_email for row in user_rows if row.user_email},
        org_alias_by_id={row.organization_id: row.organization_alias for row in org_rows if row.organization_alias},
        model_name_by_id={row.model_id: row.model_name for row in model_rows if row.model_name},
    )


def _db_object_alias(log: AuditLogResponse, aliases: _AliasMaps) -> str | None:
    if log.table_name == _KEY_TABLE:
        return aliases.key_alias_by_token.get(log.object_id)
    if log.table_name == _TEAM_TABLE:
        return aliases.team_alias_by_id.get(log.object_id)
    if log.table_name == _USER_TABLE:
        return aliases.user_alias_by_id.get(log.object_id) or aliases.user_email_by_id.get(log.object_id)
    if log.table_name == _ORG_TABLE:
        return aliases.org_alias_by_id.get(log.object_id)
    if log.table_name == _MODEL_TABLE:
        return aliases.model_name_by_id.get(log.object_id)
    return None


def _alias_from_blobs(log: AuditLogResponse, blob_keys: Tuple[str, ...]) -> str | None:
    for blob in (log.updated_values, log.before_value):
        if not isinstance(blob, dict):
            continue
        for blob_key in blob_keys:
            value = blob.get(blob_key)
            if isinstance(value, str) and value:
                return value
    return None


def _enrich_audit_log(log: AuditLogResponse, aliases: _AliasMaps) -> AuditLogResponse:
    object_alias: Final = _db_object_alias(log, aliases) or _alias_from_blobs(
        log, _BLOB_ALIAS_KEYS.get(log.table_name, ())
    )
    return log.model_copy(
        update={
            "object_alias": object_alias,
            "changed_by_user_email": aliases.user_email_by_id.get(log.changed_by),
            "changed_by_key_alias": aliases.key_alias_by_token.get(log.changed_by_api_key),
        }
    )


async def _enrich_audit_logs(
    prisma_client: "PrismaClient", audit_logs: Sequence[AuditLogResponse]
) -> List[AuditLogResponse]:
    if not audit_logs:
        return []
    aliases: Final = await _fetch_alias_maps(prisma_client, audit_logs)
    return [_enrich_audit_log(log, aliases) for log in audit_logs]


async def _build_object_team_condition(prisma_client: "PrismaClient", object_team: str) -> Dict[str, Any]:
    team_rows: Final = await prisma_client.db.litellm_teamtable.find_many(
        where={"team_alias": {"contains": object_team}}
    )
    match_values: Final = dict.fromkeys([object_team, *(row.team_id for row in team_rows)])
    return {
        "OR": [
            _build_json_field_or_condition("team_alias", object_team),
            *(_build_json_field_or_condition("team_id", value) for value in match_values),
        ]
    }


def _build_json_field_or_condition(json_key: str, value: str) -> Dict[str, Any]:
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
    changed_by: Optional[str] = Query(None, description="Filter by user or system that performed the action"),
    changed_by_api_key: Optional[str] = Query(None, description="Filter by API key hash that performed the action"),
    action: Optional[str] = Query(None, description="Filter by action type (create, update, delete)"),
    table_name: Optional[str] = Query(None, description="Filter by table name that was modified"),
    object_id: Optional[str] = Query(None, description="Filter by ID of the object that was modified"),
    start_date: Optional[str] = Query(None, description="Filter logs after this date"),
    end_date: Optional[str] = Query(None, description="Filter logs before this date"),
    object_team_id: Optional[str] = Query(
        None,
        description="Filter by team_id present in before_value or updated_values JSON (PostgreSQL only)",
    ),
    object_team: str | None = Query(
        None,
        description=(
            "Filter by team id or alias: matches team_id or team_alias present in before_value or "
            "updated_values JSON, or teams whose team_alias contains this value (PostgreSQL only)"
        ),
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

    Note: object_team_id, object_team and object_key_hash use Prisma JSON path
    filtering, which requires PostgreSQL. object_team matches a team_id or
    team_alias in the audit blobs, or any team whose team_alias contains the
    value.
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

    # JSON field filters (PostgreSQL only) — each filter is AND'd with the
    # others, but checks both before_value and updated_values internally (OR).
    if object_team_id:
        where_conditions["AND"] = where_conditions.get("AND", []) + [
            _build_json_field_or_condition("team_id", object_team_id)
        ]
    if object_key_hash:
        where_conditions["AND"] = where_conditions.get("AND", []) + [
            _build_json_field_or_condition("token", object_key_hash)
        ]
    if object_team:
        where_conditions["AND"] = where_conditions.get("AND", []) + [
            await _build_object_team_condition(prisma_client, object_team)
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

    enriched_logs: Final = await _enrich_audit_logs(
        prisma_client,
        [AuditLogResponse(**audit_log.model_dump()) for audit_log in audit_logs] if audit_logs else [],
    )

    # Return paginated response
    return PaginatedAuditLogResponse(
        audit_logs=enriched_logs,
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
async def get_audit_log_by_id(id: str, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
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
        raise HTTPException(status_code=404, detail={"message": f"Audit log with ID {id} not found"})

    enriched_logs: Final = await _enrich_audit_logs(prisma_client, [AuditLogResponse(**audit_log.model_dump())])
    return enriched_logs[0]
