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
from litellm._uuid import uuid
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.management_endpoints import (
    CACHE_SETTINGS_FIELDS,
    REDIS_TYPE_DESCRIPTIONS,
    CacheSettingsField,
)

router = APIRouter()


_REDACTED_VALUE = "***REDACTED***"


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
        verbose_proxy_logger.warning(
            "Failed to write cache-settings audit log: %s", exc
        )


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
                changed_by=litellm_changed_by
                or user_api_key_dict.user_id
                or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.CACHE_CONFIG_TABLE_NAME,
                object_id="cache_config",
                action=action,
                updated_values=json.dumps(
                    {"settings": _redact_settings(after_settings)}, default=str
                ),
                before_value=json.dumps(
                    {"settings": _redact_settings(before_settings)}, default=str
                ),
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
            cache_config = await prisma_client.db.litellm_cacheconfig.find_unique(
                where={"id": "cache_config"}
            )
            if cache_config is not None and cache_config.cache_settings:
                # Parse cache settings JSON
                cache_settings_json = cache_config.cache_settings
                if isinstance(cache_settings_json, str):
                    cache_settings_dict = json.loads(cache_settings_json)
                else:
                    cache_settings_dict = cache_settings_json

                # Decrypt cache settings
                decrypted_settings = proxy_config._decrypt_db_variables(
                    variables_dict=cache_settings_dict
                )

                # Remove redis_type if present (UI-only field, not a Cache parameter)
                # We derive it for UI in get_cache_settings endpoint
                cache_params = {
                    k: v for k, v in decrypted_settings.items() if k != "redis_type"
                }

                # Check if cache params have changed
                if (
                    CacheSettingsManager._last_cache_params is not None
                    and CacheSettingsManager._cache_params_equal(
                        CacheSettingsManager._last_cache_params, cache_params
                    )
                ):
                    verbose_proxy_logger.debug(
                        "Cache settings unchanged, skipping reinitialization"
                    )
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
    fields: List[CacheSettingsField] = Field(
        description="List of all configurable cache settings with metadata"
    )
    current_values: Dict[str, Any] = Field(
        description="Current values of cache settings"
    )
    redis_type_descriptions: Dict[str, str] = Field(
        description="Descriptions for each Redis type option"
    )


class CacheTestRequest(BaseModel):
    cache_settings: Dict[str, Any] = Field(
        description="Cache settings to test connection with"
    )


class CacheTestResponse(BaseModel):
    status: str = Field(description="Connection status: 'success' or 'failed'")
    message: str = Field(description="Connection result message")
    error: Optional[str] = Field(
        default=None, description="Error message if connection failed"
    )


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

        # Try to get cache settings from database
        current_values = {}
        if prisma_client is not None:
            cache_config = await prisma_client.db.litellm_cacheconfig.find_unique(
                where={"id": "cache_config"}
            )
            if cache_config is not None and cache_config.cache_settings:
                # Decrypt cache settings
                cache_settings_json = cache_config.cache_settings
                if isinstance(cache_settings_json, str):
                    cache_settings_dict = json.loads(cache_settings_json)
                else:
                    cache_settings_dict = cache_settings_json

                # Decrypt environment variables
                decrypted_settings = proxy_config._decrypt_db_variables(
                    variables_dict=cache_settings_dict
                )

                # Derive redis_type for UI based on settings
                # UI uses redis_type to show/hide fields, backend only stores 'type'
                if decrypted_settings.get("type") == "redis":
                    if decrypted_settings.get("redis_startup_nodes"):
                        decrypted_settings["redis_type"] = "cluster"
                    elif decrypted_settings.get("sentinel_nodes"):
                        decrypted_settings["redis_type"] = "sentinel"
                    else:
                        decrypted_settings["redis_type"] = "node"

                current_values = decrypted_settings

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
        raise HTTPException(
            status_code=500, detail=f"Error fetching cache settings: {str(e)}"
        )


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

    try:
        cache_settings = request.cache_settings.copy()
        verbose_proxy_logger.debug(
            "Testing cache connection with settings: %s", cache_settings
        )

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
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )

    try:
        cache_settings = request.cache_settings.copy()

        # Snapshot the prior settings (key set only — values get redacted in
        # the audit row) so the audit-log entry shows which fields changed.
        existing_row = await prisma_client.db.litellm_cacheconfig.find_unique(
            where={"id": "cache_config"}
        )
        before_settings: Optional[Dict[str, Any]] = None
        if existing_row is not None and existing_row.cache_settings:
            try:
                before_settings = json.loads(existing_row.cache_settings)
            except (TypeError, ValueError):
                before_settings = None
        action: AUDIT_ACTIONS = "updated" if existing_row is not None else "created"

        # Encrypt sensitive fields (keep redis_type for storage)
        encrypted_settings = proxy_config._encrypt_env_variables(
            environment_variables=cache_settings
        )

        # Save to database
        await prisma_client.db.litellm_cacheconfig.upsert(
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
        decrypted_settings = proxy_config._decrypt_db_variables(
            variables_dict=encrypted_settings
        )

        # Remove redis_type if present (UI-only field, not a Cache parameter)
        cache_params = {
            k: v for k, v in decrypted_settings.items() if k != "redis_type"
        }

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
            "settings": cache_settings,
        }
    except Exception as e:
        verbose_proxy_logger.error(f"Error updating cache settings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating cache settings: {str(e)}"
        )
