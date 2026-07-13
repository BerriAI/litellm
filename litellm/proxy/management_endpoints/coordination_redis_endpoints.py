"""
COORDINATION REDIS SETTINGS MANAGEMENT

Endpoints for managing `general_settings.coordination_redis` - the standalone
Redis the proxy uses for cross-pod coordination (tpm/rpm rate limits, spend
tracking, pod lock manager, shared health checks), configured independently of
the response-cache backend.

GET /coordination_redis/settings - Get the coordination Redis settings, field metadata, and which source is active
POST /coordination_redis/settings - Save coordination Redis settings to the database
POST /coordination_redis/settings/test - Test a coordination Redis connection with the provided credentials
"""

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.caching.caching import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    CoordinationRedisParams,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import invalidate_config_param
from litellm.repositories.config_repository import ConfigRepository
from litellm.secret_managers.main import get_secret_str
from litellm.types.management_endpoints import (
    COORDINATION_REDIS_SETTINGS_FIELDS,
    CoordinationRedisSettingsField,
    CoordinationRedisSource,
)

router = APIRouter()

_GENERAL_SETTINGS_PARAM_NAME = "general_settings"
_COORDINATION_REDIS_KEY = "coordination_redis"

# Fields that carry credentials. Redacted on read so a plaintext Redis /
# Sentinel password never leaves the server, and scrubbed out of connection-test
# error strings. `url` is here because a Redis url can embed a password inline
# (e.g. redis://:secret@host:6379/1).
_SENSITIVE_FIELDS: frozenset[str] = frozenset({"password", "sentinel_password", "url"})

_REDACTED_VALUE = "***REDACTED***"

_ENV_REF_PREFIX = "os.environ/"

_PING_TIMEOUT_SECONDS = 5.0

_SETTINGS_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


def _enforce_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins can manage coordination Redis settings"},
        )


def _resolve_env_ref(value: object) -> object:
    """Resolve an `os.environ/VAR` reference to its value, passing anything else through."""
    if isinstance(value, str) and value.startswith(_ENV_REF_PREFIX):
        return get_secret_str(value)
    return value


def _resolve_env_refs(settings: Mapping[str, object]) -> dict[str, object]:
    return {key: _resolve_env_ref(value) for key, value in settings.items()}


def _redact_credentials(settings: Mapping[str, object]) -> dict[str, object]:
    """Replace credential-bearing values with a fixed marker, keeping the rest intact."""
    return {
        key: (_REDACTED_VALUE if key in _SENSITIVE_FIELDS and value is not None else value)
        for key, value in settings.items()
    }


def _redact_all_values(settings: Optional[Mapping[str, object]]) -> dict[str, object]:
    """Replace every value with a fixed marker, preserving the key set.

    The audit row shows *which* fields changed without the audit table becoming
    a credential-harvest sink.
    """
    if not settings:
        return {}
    return {key: _REDACTED_VALUE for key in settings}


def _credential_values(settings: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(value) for key, value in settings.items() if key in _SENSITIVE_FIELDS and isinstance(value, (str, int))
    )


def _scrub_credentials(message: str, settings: Mapping[str, object]) -> str:
    """Strip any credential value the caller supplied out of an error string.

    Redis client errors routinely echo the connection url (password inline) or
    the auth error back to the caller.
    """
    scrubbed = message
    for secret in _credential_values(settings):
        if secret:
            scrubbed = scrubbed.replace(secret, _REDACTED_VALUE)
    return scrubbed


def _merge_over_saved(
    incoming: Mapping[str, object],
    saved: Mapping[str, object],
) -> dict[str, object]:
    """Restore the real credential behind every value the caller echoed back redacted.

    GET returns credentials as ``***REDACTED***``; an admin who edits the
    non-secret fields and re-submits would otherwise test (and save) the marker
    as the password.
    """
    return {
        key: (saved[key] if value == _REDACTED_VALUE and key in saved else value) for key, value in incoming.items()
    }


def _validated_params(settings: Mapping[str, object]) -> CoordinationRedisParams:
    """Validate settings the way startup does: resolve env refs, then require a connection target."""
    try:
        params = CoordinationRedisParams(**_resolve_env_refs(settings))
    except ValidationError as e:
        invalid_fields = sorted({str(error["loc"][0]) for error in e.errors() if error["loc"]})
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid coordination_redis settings for fields: {invalid_fields}"},
        )
    if not params.has_connection_target():
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "coordination_redis needs a connection target: "
                    "set one of host, url, startup_nodes, or sentinel_nodes"
                )
            },
        )
    return params


async def _read_general_settings() -> dict[str, object]:
    """Read the persisted `general_settings` config row (empty when unset or no DB)."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return {}
    config_param = await ConfigRepository(prisma_client).get_param(_GENERAL_SETTINGS_PARAM_NAME)
    if config_param is None or config_param.param_value is None:
        return {}
    return _SETTINGS_ADAPTER.validate_python(config_param.param_value)


async def get_persisted_coordination_redis_settings() -> Optional[dict[str, object]]:
    """The coordination_redis block saved to the database, if any.

    Read at startup so settings saved from the admin UI take effect on the next
    boot, and used here so a read reports what the proxy would boot with.
    """
    persisted = (await _read_general_settings()).get(_COORDINATION_REDIS_KEY)
    if isinstance(persisted, dict):
        return _SETTINGS_ADAPTER.validate_python(persisted)
    return None


async def _current_coordination_redis_settings() -> Optional[dict[str, object]]:
    """The coordination_redis block the proxy would boot with.

    The persisted row wins over the yaml-loaded config state because startup
    applies the DB `general_settings` row over the file config.
    """
    from litellm.proxy.proxy_server import proxy_config

    persisted = await get_persisted_coordination_redis_settings()
    if persisted is not None:
        return persisted

    config_state = _SETTINGS_ADAPTER.validate_python(proxy_config.get_config_state())
    general_settings = config_state.get(_GENERAL_SETTINGS_PARAM_NAME)
    if not isinstance(general_settings, dict):
        return None
    from_file = general_settings.get(_COORDINATION_REDIS_KEY)
    if isinstance(from_file, dict):
        return _SETTINGS_ADAPTER.validate_python(from_file)
    return None


def _coordination_redis_source(settings: Optional[Mapping[str, object]]) -> Optional[CoordinationRedisSource]:
    """Which source the proxy's coordination Redis comes from, in startup precedence order.

    Mirrors `ProxyConfig._init_coordination_redis` -> `ProxyConfig._init_cache`:
    an explicit block wins, else a plain-Redis response-cache backend is
    borrowed, else the REDIS_* environment fallback applies.
    """
    from litellm.proxy.proxy_server import _environment_has_redis_connection_target

    if settings:
        return "coordination_redis"
    cache_backend = litellm.cache.cache if litellm.cache is not None else None
    if isinstance(cache_backend, (RedisCache, RedisClusterCache)):
        return "cache_backend"
    if _environment_has_redis_connection_target():
        return "environment"
    return None


def _log_audit_task_exception(task: "asyncio.Task[None]") -> None:
    """Surface a fire-and-forget audit-log task failure as a warning."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        verbose_proxy_logger.warning("Failed to write coordination-redis-settings audit log: %s", exc)


async def _emit_coordination_redis_audit_log(
    *,
    action: AUDIT_ACTIONS,
    before_settings: Optional[Mapping[str, object]],
    after_settings: Optional[Mapping[str, object]],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
) -> None:
    """Emit an audit-log row for a /coordination_redis/settings mutation."""
    if litellm.store_audit_logs is not True:
        return

    from litellm.proxy.management_helpers.audit_logs import create_audit_log_for_update
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    task = asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by or user_api_key_dict.user_id or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.CONFIG_TABLE_NAME,
                object_id=_COORDINATION_REDIS_KEY,
                action=action,
                updated_values=json.dumps({"settings": _redact_all_values(after_settings)}, default=str),
                before_value=json.dumps({"settings": _redact_all_values(before_settings)}, default=str),
            )
        )
    )
    task.add_done_callback(_log_audit_task_exception)


class CoordinationRedisSettingsResponse(BaseModel):
    values: dict[str, object] = Field(description="Current coordination Redis settings, with credentials redacted")
    fields: list[CoordinationRedisSettingsField] = Field(
        description="List of all configurable coordination Redis settings with metadata"
    )
    source: Optional[CoordinationRedisSource] = Field(
        description="Where the proxy's coordination Redis comes from; null when it has none"
    )


class CoordinationRedisSettingsRequest(BaseModel):
    settings: dict[str, object] = Field(description="Coordination Redis connection params")


class CoordinationRedisTestResponse(BaseModel):
    status: str = Field(description="Connection status: 'healthy' or 'unhealthy'")
    error: Optional[str] = Field(default=None, description="Error message if the connection failed")


@router.get(
    "/coordination_redis/settings",
    tags=["Coordination Redis Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CoordinationRedisSettingsResponse,
)
async def get_coordination_redis_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CoordinationRedisSettingsResponse:
    """
    Get the coordination Redis configuration and available settings.

    Returns:
    - values: current coordination Redis settings, with password/sentinel_password/url redacted
    - fields: all configurable settings with their metadata (type, description, default, section)
    - source: "coordination_redis" | "cache_backend" | "environment" | null
    """
    _enforce_proxy_admin(user_api_key_dict)

    settings = await _current_coordination_redis_settings()
    source = _coordination_redis_source(settings)

    values = _redact_credentials(settings or {})
    fields = [field.model_copy(deep=True) for field in COORDINATION_REDIS_SETTINGS_FIELDS]
    for field in fields:
        if field.field_name in values:
            field.field_value = values[field.field_name]

    return CoordinationRedisSettingsResponse(values=values, fields=fields, source=source)


@router.post(
    "/coordination_redis/settings",
    tags=["Coordination Redis Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_coordination_redis_settings(
    request: CoordinationRedisSettingsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
) -> dict[str, object]:
    """
    Save coordination Redis settings under `general_settings.coordination_redis`.

    Parameters:
    - settings: dict - Redis connection params (host, port, username, password, url, ssl, startup_nodes, sentinel_nodes, sentinel_password, service_name). Values may be `os.environ/VAR` references, which are stored as written and resolved at startup

    The settings are written to the `general_settings` row of LiteLLM_Config,
    which startup merges over the yaml config; the proxy picks them up on its
    next restart.
    """
    from litellm.proxy.proxy_server import prisma_client, store_model_in_db

    _enforce_proxy_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected. Please connect a database."},
        )

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={"error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."},
        )

    saved_settings = await _current_coordination_redis_settings()
    settings = _merge_over_saved(request.settings, saved_settings or {})
    _validated_params(settings)

    general_settings = await _read_general_settings()
    before_settings = general_settings.get(_COORDINATION_REDIS_KEY)
    action: AUDIT_ACTIONS = "updated" if isinstance(before_settings, dict) else "created"

    await ConfigRepository(prisma_client).set_param(
        param_name=_GENERAL_SETTINGS_PARAM_NAME,
        param_value={**general_settings, _COORDINATION_REDIS_KEY: settings},
    )
    await invalidate_config_param(_GENERAL_SETTINGS_PARAM_NAME)

    # coordination_redis carries Redis credentials and decides where cross-pod
    # rate-limit and spend state lives; an admin repointing it is a
    # data-routing pivot, so make the change traceable.
    await _emit_coordination_redis_audit_log(
        action=action,
        before_settings=before_settings if isinstance(before_settings, dict) else None,
        after_settings=settings,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )

    return {
        "message": "Coordination Redis settings updated successfully. Restart the proxy to apply them.",
        "status": "success",
        "settings": _redact_credentials(settings),
    }


@router.post(
    "/coordination_redis/settings/test",
    tags=["Coordination Redis Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CoordinationRedisTestResponse,
)
async def check_coordination_redis_connection(
    request: CoordinationRedisSettingsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CoordinationRedisTestResponse:
    """
    Test a coordination Redis connection with the provided credentials.

    Parameters:
    - settings: dict - Redis connection params to test. Credential fields sent back as `***REDACTED***` fall back to the saved value

    Builds a throwaway client (never touching global state) and pings it.
    """
    from litellm.proxy.proxy_server import _build_redis_usage_cache

    _enforce_proxy_admin(user_api_key_dict)

    saved_settings = await _current_coordination_redis_settings()
    settings = _merge_over_saved(request.settings, saved_settings or {})
    params = _validated_params(settings)

    redis_cache: Optional[RedisCache] = None
    try:
        redis_cache = _build_redis_usage_cache(params.model_dump(exclude_none=True))
        await asyncio.wait_for(redis_cache.ping(), timeout=_PING_TIMEOUT_SECONDS)
        return CoordinationRedisTestResponse(status="healthy")
    except asyncio.TimeoutError:
        return CoordinationRedisTestResponse(
            status="unhealthy",
            error=f"Connection timed out after {_PING_TIMEOUT_SECONDS}s",
        )
    except Exception as e:  # noqa: BLE001  # any client/connection failure is a health verdict, not a 500
        return CoordinationRedisTestResponse(status="unhealthy", error=_scrub_credentials(str(e), settings))
    finally:
        if redis_cache is not None:
            with suppress(Exception):
                await redis_cache.disconnect()
