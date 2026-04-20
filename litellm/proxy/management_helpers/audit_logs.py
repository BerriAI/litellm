"""
Functions to create audit logs for LiteLLM Proxy
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    Optional,
    UserAPIKeyAuth,
)
from litellm.types.utils import StandardAuditLogPayload

_audit_log_callback_cache: Dict[str, CustomLogger] = {}


def _resolve_audit_log_callback(name: str) -> Optional[CustomLogger]:
    """Resolve a string callback name to a CustomLogger instance, with caching."""
    if name in _audit_log_callback_cache:
        return _audit_log_callback_cache[name]

    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
    )

    instance = _init_custom_logger_compatible_class(
        logging_integration=name,  # type: ignore
        internal_usage_cache=None,
        llm_router=None,
    )

    if instance is not None:
        _audit_log_callback_cache[name] = instance
    return instance


def _build_audit_log_payload(
    request_data: LiteLLM_AuditLogs,
) -> StandardAuditLogPayload:
    """Convert LiteLLM_AuditLogs to StandardAuditLogPayload for callback dispatch."""
    updated_at = ""
    if request_data.updated_at is not None:
        updated_at = request_data.updated_at.isoformat()

    table_name_str: str = (
        request_data.table_name.value
        if isinstance(request_data.table_name, LitellmTableNames)
        else str(request_data.table_name)
    )

    return StandardAuditLogPayload(
        id=request_data.id,
        updated_at=updated_at,
        changed_by=request_data.changed_by or "",
        changed_by_api_key=request_data.changed_by_api_key or "",
        action=request_data.action,
        table_name=table_name_str,
        object_id=request_data.object_id,
        before_value=request_data.before_value,
        updated_values=request_data.updated_values,
    )


def _audit_log_task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from audit log callback tasks so they don't slip through silently."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        verbose_proxy_logger.error(
            "Audit log callback task failed: %s", exc, exc_info=exc
        )


async def _dispatch_audit_log_to_callbacks(
    request_data: LiteLLM_AuditLogs,
) -> None:
    """Dispatch audit log to all registered audit_log_callbacks."""
    if not litellm.audit_log_callbacks:
        return

    payload = _build_audit_log_payload(request_data)

    for callback in litellm.audit_log_callbacks:
        try:
            resolved: Optional[CustomLogger] = (
                callback if isinstance(callback, CustomLogger) else None
            )
            if isinstance(callback, str):
                resolved = _resolve_audit_log_callback(callback)
                if resolved is None:
                    verbose_proxy_logger.warning(
                        "Could not resolve audit log callback: %s", callback
                    )
                    continue

            if isinstance(resolved, CustomLogger):
                task = asyncio.create_task(resolved.async_log_audit_log_event(payload))
                task.add_done_callback(_audit_log_task_done_callback)
        except Exception as e:
            verbose_proxy_logger.error(
                "Failed dispatching audit log to callback: %s", e
            )


async def create_object_audit_log(
    object_id: str,
    action: AUDIT_ACTIONS,
    litellm_changed_by: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: Optional[str],
    table_name: LitellmTableNames,
    before_value: Optional[str] = None,
    after_value: Optional[str] = None,
):
    """
    Create an audit log for an internal user.

    Parameters:
    - user_id: str - The id of the user to create the audit log for.
    - action: AUDIT_ACTIONS - The action to create the audit log for.
    - user_row: LiteLLM_UserTable - The user row to create the audit log for.
    - litellm_changed_by: Optional[str] - The user id of the user who is changing the user.
    - user_api_key_dict: UserAPIKeyAuth - The user api key dictionary.
    - litellm_proxy_admin_name: Optional[str] - The name of the proxy admin.
    """
    from litellm.secret_managers.main import get_secret_bool

    _store_audit_logs: Optional[bool] = litellm.store_audit_logs or get_secret_bool(
        "LITELLM_STORE_AUDIT_LOGS"
    )

    if _store_audit_logs is not True:
        return

    _changed_by = (
        litellm_changed_by or user_api_key_dict.user_id or litellm_proxy_admin_name
    )

    await create_audit_log_for_update(
        request_data=LiteLLM_AuditLogs(
            id=str(uuid.uuid4()),
            updated_at=datetime.now(timezone.utc),
            changed_by=_changed_by,
            changed_by_api_key=user_api_key_dict.api_key,
            table_name=table_name,
            object_id=object_id,
            action=action,
            updated_values=after_value,
            before_value=before_value,
        )
    )


async def create_audit_log_for_update(request_data: LiteLLM_AuditLogs):
    """
    Create an audit log for an object.
    """
    from litellm.secret_managers.main import get_secret_bool

    _store_audit_logs: Optional[bool] = litellm.store_audit_logs or get_secret_bool(
        "LITELLM_STORE_AUDIT_LOGS"
    )
    if _store_audit_logs is not True:
        return

    from litellm.proxy.proxy_server import premium_user, prisma_client

    if premium_user is not True:
        return

    verbose_proxy_logger.debug("creating audit log for %s", request_data)

    if isinstance(request_data.updated_values, dict):
        request_data.updated_values = json.dumps(request_data.updated_values)

    if isinstance(request_data.before_value, dict):
        request_data.before_value = json.dumps(request_data.before_value)

    # Dispatch to external audit log callbacks regardless of DB availability
    await _dispatch_audit_log_to_callbacks(request_data)

    if prisma_client is None:
        verbose_proxy_logger.error(
            "prisma_client is None, cannot write audit log to DB"
        )
        return

    _request_data = request_data.model_dump(exclude_none=True)

    try:
        await prisma_client.db.litellm_auditlog.create(
            data={
                **_request_data,  # type: ignore
            }
        )
    except Exception as e:
        # [Non-Blocking Exception. Do not allow blocking LLM API call]
        verbose_proxy_logger.error(f"Failed Creating audit log {e}")
