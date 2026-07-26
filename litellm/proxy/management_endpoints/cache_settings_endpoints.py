"""
CACHE SETTINGS MANAGEMENT

Endpoints for managing cache configuration

GET /cache/settings - Get cache configuration including available settings
POST /cache/settings/test - Test cache connection with provided credentials
POST /cache/settings - Save cache settings to database
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._redis import _redis_kwargs_from_environment
from litellm._uuid import uuid
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import call_with_db_reconnect_retry
from litellm.repositories.table_repositories import CacheConfigRepository
from litellm.types.management_endpoints import (
    CACHE_SETTINGS_FIELDS,
    REDIS_TYPE_DESCRIPTIONS,
    CacheSettingsField,
)

router = APIRouter()

# Cache fields holding credentials. Masked on read so plaintext Redis /
# Sentinel passwords never leave the server in a GET response. `url` is here
# because a Redis/Valkey URL can embed a password inline
# (e.g. redis://:secret@host:6379/1).
_CACHE_SENSITIVE_FIELDS: set = {"password", "sentinel_password", "url"}

# The env fallback resolves the full set of redis.Redis kwargs, which includes
# credential-bearing params (azure_client_secret, ssl_password, ...) that are
# not cache UI fields. Only overlay fields the settings page actually renders,
# so the read never surfaces a credential the UI does not manage.
_CACHE_SETTINGS_FIELD_NAMES: frozenset = frozenset(field.field_name for field in CACHE_SETTINGS_FIELDS)

# Classifier used, alongside _CACHE_SENSITIVE_FIELDS, to redact any
# credential-bearing key before it leaves the server (`url` is kept in the
# explicit set because its name carries no sensitive segment).
_CREDENTIAL_CLASSIFIER = SensitiveDataMasker()


_REDACTED_VALUE = "***REDACTED***"


_URL_OVERRIDDEN_CONNECTION_FIELDS: frozenset = frozenset({"host", "port", "db", "password", "username"})


def _resolve_cache_url_precedence(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return cache settings with the url-vs-discrete-fields ambiguity resolved.

    When a full ``url`` is supplied it wins: the discrete
    host/port/db/password/username fields are dropped so the persisted config
    is unambiguous and matches runtime resolution in ``litellm._redis``
    (``redis.Redis.from_url`` ignores them). Cluster mode
    (``redis_startup_nodes``) is exempt because it authenticates via the
    discrete fields rather than a url.
    """
    url = settings.get("url")
    has_url = isinstance(url, str) and url.strip() != ""
    if not has_url or settings.get("redis_startup_nodes"):
        return dict(settings)
    return {k: v for k, v in settings.items() if k not in _URL_OVERRIDDEN_CONNECTION_FIELDS}


def _parse_stored_settings(cache_settings_value: object) -> dict[str, Any]:
    """Normalize a stored cache_settings blob to a dict.

    The prisma column comes back as either a JSON string or an already-parsed
    dict depending on the client, so callers that json.loads unconditionally
    silently drop the whole (still-encrypted) row on the dict path.
    """
    parsed = json.loads(cache_settings_value) if isinstance(cache_settings_value, str) else cache_settings_value
    return parsed if isinstance(parsed, dict) else {}


def _overlay_environment(stored: Mapping[str, Any]) -> dict[str, Any]:
    """Fill connection fields from the REDIS_* environment the cache actually reads.

    A response cache pointed at Redis resolves host/port/password/etc. from the
    REDIS_* env vars when the stored config leaves them unset, so a cache
    configured purely through the environment works while its settings page,
    which reads only the database row, shows blank. Overlaying the same env
    kwargs the runtime uses makes the page reflect the effective connection.
    Stored values win; the environment only fills what the stored config omits.
    """
    env_kwargs = {
        key: value for key, value in _redis_kwargs_from_environment().items() if key in _CACHE_SETTINGS_FIELD_NAMES
    }
    if not env_kwargs:
        return dict(stored)
    effective = {**env_kwargs, **stored}
    # the env fallback is a Redis connection, so name the type when the stored
    # config did not, letting the UI render the Redis fields it just populated
    effective.setdefault("type", "redis")
    return effective


def _redact_credentials(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Replace credential-bearing values with a fixed marker, keeping the rest.

    The marker is unambiguous on the way back in: an admin who edits an
    unrelated field and re-submits sends the marker for the untouched secret,
    which the update path maps back to the stored value rather than persisting
    the marker over a working password.
    """
    return {
        key: (_REDACTED_VALUE if value is not None and _is_credential_field(key) else value)
        for key, value in settings.items()
    }


def _is_credential_field(key: str) -> bool:
    """Whether a cache setting carries a credential and must be redacted on read."""
    return key in _CACHE_SENSITIVE_FIELDS or _CREDENTIAL_CLASSIFIER.is_sensitive_key(key)


def _has_connection_target(value: object) -> bool:
    """Whether a payload value names a live discrete connection target."""
    if isinstance(value, str):
        return value.strip() != "" and value != _REDACTED_VALUE
    return value not in (None, [], {})


# Every field that identifies which Redis a credential belongs to, across node
# (host/port/url), cluster (redis_startup_nodes), and sentinel
# (sentinel_nodes/service_name) modes. A stored secret is bound to these.
_CONNECTION_TARGET_FIELDS: tuple = (
    "host",
    "port",
    "url",
    "redis_startup_nodes",
    "sentinel_nodes",
    "service_name",
)


def _target_repr(value: object) -> str:
    """Canonical string form of a connection-target value for equality checks.

    The client may serialize the same target differently from storage (a port as
    "6379" vs 6379, node lists round-tripped through JSON), so compare normalized
    forms rather than raw values to avoid treating an unchanged target as a change.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _saved_secret_is_reusable(incoming: Mapping[str, Any], saved: Mapping[str, Any]) -> bool:
    """Whether a stored credential may be restored for this request.

    A stored secret belongs to the stored connection target, so it is reused only
    when the request describes that same target on every dimension the stored
    config pins (host/port, url, cluster nodes, sentinel nodes/service). This
    prevents credential replay: a caller cannot omit the credential, point at a
    different (or incomplete) target, and have the proxy send the stored secret
    to a Redis of their choosing.

    Non-secret target fields (host/port/nodes/service) must be supplied and match
    in normalized form, so equivalent representations (port "6379" vs 6379) are
    not seen as a change while an omitted or different value is. ``url`` is the
    exception: it is itself the secret and the form never re-prefills it, so a
    redacted or omitted url means "keep the stored url" (same target) and only a
    different supplied url blocks reuse.
    """
    for field in _CONNECTION_TARGET_FIELDS:
        saved_value = saved.get(field)
        if saved_value in (None, "", [], {}):
            continue  # the stored config does not pin this dimension
        incoming_value = incoming.get(field)
        if field == "url":
            if incoming_value in (None, "", _REDACTED_VALUE):
                continue  # url kept as-is (same target)
            if _target_repr(incoming_value) != _target_repr(saved_value):
                return False
            continue
        if _target_repr(incoming_value) != _target_repr(saved_value):
            return False  # a pinned target field is missing or different
    return True


def _merge_over_saved(incoming: Mapping[str, Any], saved: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the stored secret behind any credential the caller echoed back redacted or omitted.

    GET returns credentials as the marker and the form never re-prefills a
    secret, so a save that does not touch a credential arrives with the marker
    or with the field absent. Either way the real secret must survive: it is
    restored from the stored row, or dropped when there is no stored row (the
    value is env-sourced and the marker must never be persisted). Non-secret
    fields are taken from the incoming payload as-is, so clearing one still works.

    ``url`` is the exception: it is credential-bearing (redacted) yet also a
    connection-mode selector that url-precedence resolves against host/port. If
    the caller supplies a discrete target (host, cluster, or sentinel nodes), a
    stored url is a stale mode the caller is leaving, so it is dropped rather
    than restored, otherwise url-precedence would resurrect it and discard the
    submitted host/port.
    """
    switching_to_discrete_target = (
        _has_connection_target(incoming.get("host"))
        or _has_connection_target(incoming.get("redis_startup_nodes"))
        or _has_connection_target(incoming.get("sentinel_nodes"))
    )
    reuse_saved_secret = _saved_secret_is_reusable(incoming, saved)
    merged = dict(incoming)
    for field in _CACHE_SENSITIVE_FIELDS:
        # A value the caller explicitly supplied is honored verbatim: a new
        # secret, or an empty string / null to clear the stored one. Only an
        # omitted field or the echoed-back marker triggers preserve-or-drop.
        if field in incoming and incoming[field] != _REDACTED_VALUE:
            continue
        if field == "url" and switching_to_discrete_target:
            merged.pop(field, None)
            continue
        if field in saved and reuse_saved_secret:
            merged[field] = saved[field]
        else:
            # nothing stored to reuse, or the caller is pointing at a different
            # target: never persist/replay the marker or the stored secret
            merged.pop(field, None)
    return merged


def _redact_settings(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Replace every value in a settings map with a fixed marker.

    Cache config carries Redis credentials (passwords, connection strings).
    The audit-log row preserves the field names so a reader can see *which*
    fields changed, but values are stripped so the audit table can't itself
    become a credential-harvest sink.
    """
    if not settings:
        return {}
    return {k: _REDACTED_VALUE for k in settings.keys()}


def _log_audit_task_exception(task: "asyncio.Task[None]") -> None:
    """Surface a fire-and-forget audit-log task failure as a warning.

    ``asyncio.create_task`` swallows exceptions silently — if the audit
    write fails we'd otherwise lose the row without any signal.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        verbose_proxy_logger.warning("Failed to write cache-settings audit log: %s", exc)


async def _emit_cache_settings_audit_log(
    *,
    action: AUDIT_ACTIONS,
    before_settings: Optional[Mapping[str, Any]],
    after_settings: Optional[Mapping[str, Any]],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
) -> None:
    """Emit an audit-log row for a /cache/settings mutation.

    Mirrors the ``store_audit_logs``-gated pattern used in
    ``team_callback_endpoints.py``: fire-and-forget, no-op when audit
    logging is disabled, with a done-callback that surfaces any task
    exception.  Captured under ``LiteLLM_CacheConfig`` so the row
    co-locates with the table it mutates.
    """
    if litellm.store_audit_logs is not True:
        return

    from litellm.proxy.management_helpers.audit_logs import (
        create_audit_log_for_update,
    )
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    task = asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by or user_api_key_dict.user_id or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.CACHE_CONFIG_TABLE_NAME,
                object_id="cache_config",
                action=action,
                updated_values=json.dumps({"settings": _redact_settings(after_settings)}, default=str),
                before_value=json.dumps({"settings": _redact_settings(before_settings)}, default=str),
            )
        )
    )
    task.add_done_callback(_log_audit_task_exception)


class CacheSettingsManager:
    """
    Manages cache settings initialization and updates.
    Tracks last cache params to avoid unnecessary reinitialization.
    """

    _last_cache_params: Optional[Dict[str, Any]] = None

    @staticmethod
    def _cache_params_equal(params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """
        Compare two cache parameter dictionaries for equality.
        Normalizes values and filters out UI-only fields.
        """

        # Normalize by removing None values and UI-only fields
        def normalize(params: Dict[str, Any]) -> Dict[str, Any]:
            normalized = {}
            for k, v in params.items():
                if k == "redis_type":  # Skip UI-only field
                    continue
                if v is not None:
                    # Convert to string for comparison to handle different types
                    normalized[k] = str(v) if not isinstance(v, (list, dict)) else v
            return normalized

        normalized1 = normalize(params1)
        normalized2 = normalize(params2)

        return normalized1 == normalized2

    @staticmethod
    async def init_cache_settings_in_db(prisma_client, proxy_config):
        """
        Initialize cache settings from database into the router on startup.
        Only reinitializes if cache params have changed.
        """
        import json

        try:
            cache_config = await call_with_db_reconnect_retry(
                prisma_client,
                lambda: CacheConfigRepository(prisma_client).table.find_unique(where={"id": "cache_config"}),
                reason="init_cache_settings_in_db_lookup_failure",
            )
            if cache_config is not None and cache_config.cache_settings:
                # Parse cache settings JSON
                cache_settings_json = cache_config.cache_settings
                if isinstance(cache_settings_json, str):
                    cache_settings_dict = json.loads(cache_settings_json)
                else:
                    cache_settings_dict = cache_settings_json

                # Decrypt cache settings
                decrypted_settings = proxy_config._decrypt_db_variables(variables_dict=cache_settings_dict)

                # Remove redis_type if present (UI-only field, not a Cache parameter)
                # We derive it for UI in get_cache_settings endpoint
                cache_params = {k: v for k, v in decrypted_settings.items() if k != "redis_type"}

                # Check if cache params have changed
                if CacheSettingsManager._last_cache_params is not None and CacheSettingsManager._cache_params_equal(
                    CacheSettingsManager._last_cache_params, cache_params
                ):
                    verbose_proxy_logger.debug("Cache settings unchanged, skipping reinitialization")
                    return

                # Initialize cache only if params changed or cache not initialized
                proxy_config._init_cache(cache_params=cache_params)

                # Store the params we just initialized
                CacheSettingsManager._last_cache_params = cache_params.copy()

                # Switch on LLM response caching
                proxy_config.switch_on_llm_response_caching()

                verbose_proxy_logger.info("Cache settings initialized from database")
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.management_endpoints.cache_settings_endpoints.py::CacheSettingsManager::init_cache_settings_in_db - {}".format(
                    str(e)
                )
            )

    @staticmethod
    def update_cache_params(cache_params: Dict[str, Any]):
        """
        Update the last cache params after initialization.
        Called after cache settings are updated via the API.
        """
        CacheSettingsManager._last_cache_params = cache_params.copy()


class CacheSettingsResponse(BaseModel):
    fields: List[CacheSettingsField] = Field(description="List of all configurable cache settings with metadata")
    current_values: Dict[str, Any] = Field(description="Current values of cache settings")
    redis_type_descriptions: Dict[str, str] = Field(description="Descriptions for each Redis type option")


class CacheTestRequest(BaseModel):
    cache_settings: Dict[str, Any] = Field(description="Cache settings to test connection with")


class CacheTestResponse(BaseModel):
    status: str = Field(description="Connection status: 'success' or 'failed'")
    message: str = Field(description="Connection result message")
    error: Optional[str] = Field(default=None, description="Error message if connection failed")


class CacheSettingsUpdateRequest(BaseModel):
    cache_settings: Dict[str, Any] = Field(description="Cache settings to save")


@router.get(
    "/cache/settings",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheSettingsResponse,
)
async def get_cache_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get cache configuration and available settings.

    Returns:
    - fields: List of all configurable cache settings with their metadata (type, description, default, options)
    - current_values: Current values of cache settings from database
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    try:
        # Get cache settings fields from types file
        cache_fields = [field.model_copy(deep=True) for field in CACHE_SETTINGS_FIELDS]

        # Read the stored settings (decrypted); an env-only cache has none.
        stored: dict[str, Any] = {}
        if prisma_client is not None:
            cache_config = await CacheConfigRepository(prisma_client).table.find_unique(where={"id": "cache_config"})
            if cache_config is not None and cache_config.cache_settings:
                stored = proxy_config._decrypt_db_variables(
                    variables_dict=_parse_stored_settings(cache_config.cache_settings)
                )

        # Fill connection fields from the REDIS_* environment the cache resolves
        # from when the stored config leaves them unset, then apply url precedence
        # so a url-mode config does not surface conflicting discrete fields (which
        # would otherwise let a no-op save silently switch it to host/port).
        effective = _resolve_cache_url_precedence(_overlay_environment(stored))

        # Derive redis_type for UI based on settings
        # UI uses redis_type to show/hide fields, backend only stores 'type'
        if effective.get("type") == "redis":
            if effective.get("redis_startup_nodes"):
                effective["redis_type"] = "cluster"
            elif effective.get("sentinel_nodes"):
                effective["redis_type"] = "sentinel"
            else:
                effective["redis_type"] = "node"

        # Redact credential fields so the GET response never carries a plaintext
        # Redis / Sentinel password off the server.
        current_values = _redact_credentials(effective)

        # Update field values with current values
        for field in cache_fields:
            if field.field_name in current_values:
                field.field_value = current_values[field.field_name]

        return CacheSettingsResponse(
            fields=cache_fields,
            current_values=current_values,
            redis_type_descriptions=REDIS_TYPE_DESCRIPTIONS,
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Error fetching cache settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching cache settings: {str(e)}")


@router.post(
    "/cache/settings/test",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheTestResponse,
)
async def test_cache_connection(
    request: CacheTestRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test cache connection with provided credentials.

    Creates a temporary cache instance and uses its test_connection method
    to verify the credentials work without affecting global state.
    """
    from litellm import Cache
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    try:
        # A credential the form left untouched arrives redacted; resolve it back
        # to the stored secret so the test connects with the real password. A
        # lookup failure must not block the test, so fall back to no stored row.
        saved_settings: dict[str, Any] = {}
        if prisma_client is not None:
            try:
                existing_row = await CacheConfigRepository(prisma_client).table.find_unique(
                    where={"id": "cache_config"}
                )
                if existing_row is not None and existing_row.cache_settings:
                    saved_settings = proxy_config._decrypt_db_variables(
                        variables_dict=_parse_stored_settings(existing_row.cache_settings)
                    )
            except Exception:  # noqa: BLE001 - a saved-settings lookup failure must not block a connection test
                saved_settings = {}
        cache_settings = _resolve_cache_url_precedence(_merge_over_saved(request.cache_settings, saved_settings))
        # cache_settings now carries the resolved plaintext credential; never log it raw
        verbose_proxy_logger.debug("Testing cache connection with settings: %s", _redact_credentials(cache_settings))

        # Only support Redis for now
        if cache_settings.get("type") != "redis":
            return CacheTestResponse(
                status="failed",
                message="Only Redis cache type is currently supported for testing",
            )

        # Create temporary cache instance
        temp_cache = Cache(**cache_settings)

        # Use the cache's test_connection method
        result = await temp_cache.cache.test_connection()

        return CacheTestResponse(**result)

    except Exception as e:
        verbose_proxy_logger.error(f"Error testing cache connection: {str(e)}")
        return CacheTestResponse(
            status="failed",
            message=f"Cache connection test failed: {str(e)}",
            error=str(e),
        )


@router.post(
    "/cache/settings",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cache_settings(
    request: CacheSettingsUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Save cache settings to database and initialize cache.

    This endpoint:
    1. Encrypts sensitive fields (passwords, etc.)
    2. Saves to LiteLLM_CacheConfig table
    3. Reinitializes cache with new settings
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

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

    try:
        # Read the stored row first: its decrypted values back any credential the
        # caller echoed back redacted, and its key set drives the audit diff.
        existing_row = await CacheConfigRepository(prisma_client).table.find_unique(where={"id": "cache_config"})
        before_settings: Optional[Dict[str, Any]] = None
        saved_settings: dict[str, Any] = {}
        if existing_row is not None and existing_row.cache_settings:
            before_settings = _parse_stored_settings(existing_row.cache_settings)
            saved_settings = proxy_config._decrypt_db_variables(variables_dict=before_settings)
        action: AUDIT_ACTIONS = "updated" if existing_row is not None else "created"

        # Preserve stored secrets behind any redacted or omitted credential, then
        # resolve the url-vs-discrete-fields precedence.
        cache_settings = _resolve_cache_url_precedence(_merge_over_saved(request.cache_settings, saved_settings))

        # Encrypt sensitive fields (keep redis_type for storage)
        encrypted_settings = proxy_config._encrypt_env_variables(environment_variables=cache_settings)

        # Save to database
        await CacheConfigRepository(prisma_client).table.upsert(
            where={"id": "cache_config"},
            data={
                "create": {
                    "id": "cache_config",
                    "cache_settings": json.dumps(encrypted_settings),
                },
                "update": {
                    "cache_settings": json.dumps(encrypted_settings),
                },
            },
        )

        # Reinitialize cache with new settings
        # Decrypt for initialization
        decrypted_settings = proxy_config._decrypt_db_variables(variables_dict=encrypted_settings)

        # Remove redis_type if present (UI-only field, not a Cache parameter)
        cache_params = {k: v for k, v in decrypted_settings.items() if k != "redis_type"}

        # Initialize cache (frontend sends type="redis", not redis_type)
        proxy_config._init_cache(cache_params=cache_params)

        # Update the last cache params to avoid reinitializing unnecessarily
        CacheSettingsManager.update_cache_params(cache_params)

        # Switch on LLM response caching
        proxy_config.switch_on_llm_response_caching()

        # Cache settings carry Redis credentials and connection strings that
        # control where LLM responses are cached.  An admin (or compromised
        # admin) flipping the cache backend silently is a data-routing
        # pivot; emit an audit-log row so the action is traceable.
        await _emit_cache_settings_audit_log(
            action=action,
            before_settings=before_settings,
            after_settings=cache_settings,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

        return {
            "message": "Cache settings updated successfully",
            "status": "success",
            "settings": _redact_credentials(cache_settings),
        }
    except Exception as e:
        verbose_proxy_logger.error(f"Error updating cache settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating cache settings: {str(e)}")
