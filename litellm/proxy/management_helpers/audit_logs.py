"""
Functions to create audit logs for LiteLLM Proxy

New rows are stamped with denormalized alias columns (object_alias, object_team_id,
object_team_alias, changed_by_user_email, changed_by_key_alias) at write time. Rows
written before those columns existed keep NULLs until an operator runs the optional
db_scripts/backfill_audit_log_aliases.sql runbook.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    UserAPIKeyAuth,
)
from litellm.repositories.table_repositories import AuditLogRepository
from litellm.types.utils import StandardAuditLogPayload

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_audit_log_callback_cache: Final[dict[str, CustomLogger]] = {}
ALLOW_LITELLM_CHANGED_BY_HEADER_METADATA_KEY: Final = "allow_litellm_changed_by_header"


def _allows_litellm_changed_by_header(user_api_key_dict: UserAPIKeyAuth) -> bool:
    for admin_metadata in (user_api_key_dict.metadata, user_api_key_dict.team_metadata):
        if (
            isinstance(admin_metadata, dict)
            and admin_metadata.get(ALLOW_LITELLM_CHANGED_BY_HEADER_METADATA_KEY) is True
        ):
            return True
    return False


def get_audit_log_changed_by(
    *,
    litellm_changed_by: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
) -> str | None:
    if litellm_changed_by and _allows_litellm_changed_by_header(user_api_key_dict):
        return litellm_changed_by
    return user_api_key_dict.user_id or litellm_proxy_admin_name


class AuditLogActorFields(NamedTuple):
    changed_by_user_email: str | None
    changed_by_key_alias: str | None


def get_audit_log_actor_fields(
    *,
    litellm_changed_by: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
) -> AuditLogActorFields:
    """Actor fields from the authenticated credential. The email is only attributed when the
    resolved changed_by IS the credential's user, so a litellm-changed-by header value is
    never decorated with a real user's email."""
    changed_by: Final = get_audit_log_changed_by(
        litellm_changed_by=litellm_changed_by,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )
    is_credential_user: Final = user_api_key_dict.user_id is not None and changed_by == user_api_key_dict.user_id
    return AuditLogActorFields(
        changed_by_user_email=(user_api_key_dict.user_email or None) if is_credential_user else None,
        changed_by_key_alias=user_api_key_dict.key_alias or None,
    )


def _resolve_audit_log_callback(name: str) -> CustomLogger | None:
    """Resolve a string callback name to a CustomLogger instance, with caching.

    For "s3_v2" with `litellm.s3_audit_callback_params` set, constructs a
    dedicated `S3Logger` so audit logs can target a different bucket than the
    normal-log singleton served by `_init_custom_logger_compatible_class`.
    """
    if name in _audit_log_callback_cache:
        return _audit_log_callback_cache[name]

    instance: CustomLogger | None
    if name == "s3_v2" and getattr(litellm, "s3_audit_callback_params", None) is not None:
        from litellm.integrations.s3_v2 import S3Logger as S3V2Logger

        instance = S3V2Logger(s3_callback_params_override=litellm.s3_audit_callback_params)
    else:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        instance = _init_custom_logger_compatible_class(
            logging_integration=name,
            internal_usage_cache=None,
            llm_router=None,
        )

    if instance is not None:
        _audit_log_callback_cache[name] = instance
    return instance


def reset_audit_log_callback_cache() -> None:
    """Clear cached audit-log callback instances. Call on config reload."""
    _audit_log_callback_cache.clear()


def _build_audit_log_payload(
    request_data: LiteLLM_AuditLogs,
) -> StandardAuditLogPayload:
    """Convert LiteLLM_AuditLogs to StandardAuditLogPayload for callback dispatch."""
    updated_at = ""
    if request_data.updated_at is not None:
        updated_at = request_data.updated_at.isoformat()

    table_name_str: Final[str] = (
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
        object_alias=request_data.object_alias,
        object_team_id=request_data.object_team_id,
        object_team_alias=request_data.object_team_alias,
        changed_by_user_email=request_data.changed_by_user_email,
        changed_by_key_alias=request_data.changed_by_key_alias,
    )


def _audit_log_task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from audit log callback tasks so they don't slip through silently."""
    try:
        exc: Final = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        verbose_proxy_logger.error("Audit log callback task failed: %s", exc, exc_info=exc)


async def _dispatch_audit_log_to_callbacks(
    request_data: LiteLLM_AuditLogs,
) -> None:
    """Dispatch audit log to all registered audit_log_callbacks."""
    if not litellm.audit_log_callbacks:
        return

    payload: Final = _build_audit_log_payload(request_data)

    for callback in litellm.audit_log_callbacks:
        try:
            resolved: CustomLogger | None = callback if isinstance(callback, CustomLogger) else None
            if isinstance(callback, str):
                resolved = _resolve_audit_log_callback(callback)
                if resolved is None:
                    verbose_proxy_logger.warning("Could not resolve audit log callback: %s", callback)
                    continue

            if isinstance(resolved, CustomLogger):
                task = asyncio.create_task(resolved.async_log_audit_log_event(payload))
                task.add_done_callback(_audit_log_task_done_callback)
        except Exception as e:
            verbose_proxy_logger.error("Failed dispatching audit log to callback: %s", e)


class _AuditBlobAliasFields(BaseModel):
    team_alias: str | None = None
    user_alias: str | None = None
    user_email: str | None = None
    model_name: str | None = None
    team_id: str | None = None


def _blob_alias_fields(value: object) -> _AuditBlobAliasFields:
    try:
        if isinstance(value, dict):
            return _AuditBlobAliasFields.model_validate(value)
        if isinstance(value, str):
            return _AuditBlobAliasFields.model_validate_json(value)
    except ValidationError:
        return _AuditBlobAliasFields()
    return _AuditBlobAliasFields()


def _derive_object_alias(table_name: str, updated: _AuditBlobAliasFields, before: _AuditBlobAliasFields) -> str | None:
    if table_name == LitellmTableNames.TEAM_TABLE_NAME.value:
        return updated.team_alias or before.team_alias or None
    if table_name == LitellmTableNames.USER_TABLE_NAME.value:
        return updated.user_alias or updated.user_email or before.user_alias or before.user_email or None
    if table_name == LitellmTableNames.PROXY_MODEL_TABLE_NAME.value:
        return updated.model_name or before.model_name or None
    return None


async def _lookup_team_alias(prisma_client: "PrismaClient", team_id: str) -> str | None:
    try:
        row: Final = await prisma_client.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    except Exception as e:  # noqa: BLE001  # audit enrichment must never break the management call
        verbose_proxy_logger.debug("audit log team alias lookup failed: %s", e)
        return None
    alias: Final = None if row is None else row.team_alias
    return alias if isinstance(alias, str) and alias else None


async def _lookup_user_email(prisma_client: "PrismaClient", user_id: str) -> str | None:
    try:
        row: Final = await prisma_client.db.litellm_usertable.find_unique(where={"user_id": user_id})
    except Exception as e:  # noqa: BLE001  # audit enrichment must never break the management call
        verbose_proxy_logger.debug("audit log user email lookup failed: %s", e)
        return None
    email: Final = None if row is None else row.user_email
    return email if isinstance(email, str) and email else None


class _ActorKey(NamedTuple):
    key_alias: str | None
    user_id: str | None


async def _lookup_actor_key(prisma_client: "PrismaClient", token: str) -> _ActorKey:
    try:
        row: Final = await prisma_client.db.litellm_verificationtoken.find_unique(where={"token": token})
    except Exception as e:  # noqa: BLE001  # audit enrichment must never break the management call
        verbose_proxy_logger.debug("audit log key lookup failed: %s", e)
        return _ActorKey(key_alias=None, user_id=None)
    alias: Final = None if row is None else row.key_alias
    user_id: Final = None if row is None else row.user_id
    return _ActorKey(
        key_alias=alias if isinstance(alias, str) and alias else None,
        user_id=user_id if isinstance(user_id, str) and user_id else None,
    )


async def _lookup_key_alias(prisma_client: "PrismaClient", token: str) -> str | None:
    actor_key: Final = await _lookup_actor_key(prisma_client, token)
    return actor_key.key_alias


async def _with_denormalized_aliases(
    request_data: LiteLLM_AuditLogs, prisma_client: "PrismaClient | None"
) -> LiteLLM_AuditLogs:
    updated: Final = _blob_alias_fields(request_data.updated_values)
    before: Final = _blob_alias_fields(request_data.before_value)
    table_name: Final = (
        request_data.table_name.value
        if isinstance(request_data.table_name, LitellmTableNames)
        else str(request_data.table_name)
    )
    is_team_row: Final = table_name == LitellmTableNames.TEAM_TABLE_NAME.value
    is_key_row: Final = table_name == LitellmTableNames.KEY_TABLE_NAME.value
    fields_set: Final = request_data.model_fields_set
    changed_by: Final = request_data.changed_by if isinstance(request_data.changed_by, str) else None
    object_alias: Final = (
        request_data.object_alias
        if "object_alias" in fields_set
        else (
            await _lookup_key_alias(prisma_client, request_data.object_id)
            if is_key_row and prisma_client is not None
            else _derive_object_alias(table_name, updated, before)
        )
    )
    object_team_id: Final = (
        request_data.object_team_id
        or updated.team_id
        or before.team_id
        or (request_data.object_id if is_team_row else None)
        or None
    )
    object_team_alias: Final = (
        request_data.object_team_alias
        or updated.team_alias
        or before.team_alias
        or (object_alias if is_team_row else None)
        or (
            await _lookup_team_alias(prisma_client, object_team_id)
            if prisma_client is not None and object_team_id
            else None
        )
    )
    need_actor_key: Final = (
        prisma_client is not None
        and bool(request_data.changed_by_api_key)
        and (not request_data.changed_by_key_alias or not request_data.changed_by_user_email)
    )
    actor_key: Final = (
        await _lookup_actor_key(prisma_client, request_data.changed_by_api_key)
        if need_actor_key and prisma_client is not None and request_data.changed_by_api_key
        else _ActorKey(key_alias=None, user_id=None)
    )
    changed_by_key_alias: Final = request_data.changed_by_key_alias or actor_key.key_alias
    changed_by_user_email: Final = request_data.changed_by_user_email or (
        await _lookup_user_email(prisma_client, changed_by)
        if prisma_client is not None and changed_by is not None and actor_key.user_id == changed_by
        else None
    )
    return request_data.model_copy(
        update={
            "object_alias": object_alias,
            "object_team_id": object_team_id,
            "object_team_alias": object_team_alias,
            "changed_by_user_email": changed_by_user_email,
            "changed_by_key_alias": changed_by_key_alias,
        }
    )


async def create_object_audit_log(
    object_id: str,
    action: AUDIT_ACTIONS,
    litellm_changed_by: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
    table_name: LitellmTableNames,
    before_value: str | None = None,
    after_value: str | None = None,
    object_alias: str | None = None,
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
    - object_alias: Optional[str] - Alias of the audited object when the caller already holds it.
    """
    from litellm.secret_managers.main import get_secret_bool

    _store_audit_logs: Final[bool | None] = litellm.store_audit_logs or get_secret_bool("LITELLM_STORE_AUDIT_LOGS")

    if _store_audit_logs is not True:
        return

    _changed_by: Final = get_audit_log_changed_by(
        litellm_changed_by=litellm_changed_by,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )
    _actor: Final = get_audit_log_actor_fields(
        litellm_changed_by=litellm_changed_by,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )

    base_request: Final = LiteLLM_AuditLogs(
        id=str(uuid.uuid4()),
        updated_at=datetime.now(timezone.utc),
        changed_by=_changed_by,
        changed_by_api_key=user_api_key_dict.api_key,
        table_name=table_name,
        object_id=object_id,
        action=action,
        updated_values=after_value,
        before_value=before_value,
        changed_by_user_email=_actor.changed_by_user_email,
        changed_by_key_alias=_actor.changed_by_key_alias,
    )
    request: Final = (
        base_request.model_copy(update={"object_alias": object_alias}) if object_alias is not None else base_request
    )

    await create_audit_log_for_update(request_data=request)


async def create_audit_log_for_update(request_data: LiteLLM_AuditLogs):
    """
    Create an audit log for an object.
    """
    from litellm.secret_managers.main import get_secret_bool

    _store_audit_logs: Final[bool | None] = litellm.store_audit_logs or get_secret_bool("LITELLM_STORE_AUDIT_LOGS")
    if _store_audit_logs is not True:
        return

    from litellm.proxy.proxy_server import premium_user, prisma_client

    if premium_user is not True:
        return

    verbose_proxy_logger.debug("creating audit log for %s", request_data)

    enriched: Final = await _with_denormalized_aliases(request_data, prisma_client)

    # Dispatch to external audit log callbacks regardless of DB availability
    await _dispatch_audit_log_to_callbacks(enriched)

    if prisma_client is None:
        verbose_proxy_logger.error("prisma_client is None, cannot write audit log to DB")
        return

    _request_data: Final = enriched.model_dump(exclude_none=True)

    try:
        await AuditLogRepository(prisma_client).table.create(
            data={
                **_request_data,
            }
        )
    except Exception as e:
        # [Non-Blocking Exception. Do not allow blocking LLM API call]
        verbose_proxy_logger.error(
            "Failed creating audit log row %s (audit logs are NOT being stored; "
            "an unmigrated DB missing the alias columns causes this): %s",
            enriched.id,
            e,
            exc_info=e,
        )
