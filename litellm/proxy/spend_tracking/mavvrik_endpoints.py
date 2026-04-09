"""FastAPI admin endpoints for the Mavvrik integration.

Endpoints (all require PROXY_ADMIN role):
    POST   /mavvrik/init          Store encrypted settings in LiteLLM_Config
    GET    /mavvrik/settings      View current settings (API key masked)
    PUT    /mavvrik/settings      Update existing settings
    DELETE /mavvrik/delete        Remove all Mavvrik settings
    POST   /mavvrik/dry-run       Preview CSV records without uploading
    POST   /mavvrik/export        Trigger manual upload to GCS
"""

import json
import os
from datetime import datetime, timedelta
from datetime import timezone as _tz
from typing import Optional

import litellm
from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    MAVVRIK_EXPORT_INTERVAL_MINUTES,
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
)
from litellm.integrations.mavvrik.mavvrik import MavvrikLogger
from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.proxy.mavvrik_endpoints import (
    MavvrikDeleteResponse,
    MavvrikDryRunResponse,
    MavvrikExportRequest,
    MavvrikExportResponse,
    MavvrikInitRequest,
    MavvrikInitResponse,
    MavvrikSettingsUpdate,
    MavvrikSettingsView,
)

router = APIRouter()

# Key used in LiteLLM_Config table
_CONFIG_KEY = "mavvrik_settings"

_sensitive_masker = SensitiveDataMasker()


# ------------------------------------------------------------------
# Internal settings helpers
# ------------------------------------------------------------------


async def _set_mavvrik_settings(
    api_key: str,
    api_endpoint: str,
    connection_id: str,
    timezone: str,
    marker: Optional[str] = None,
) -> None:
    """Encrypt API key and upsert all settings into LiteLLM_Config."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    encrypted_api_key = encrypt_value_helper(api_key)
    settings: dict = {
        "api_key": encrypted_api_key,
        "api_endpoint": api_endpoint,
        "connection_id": connection_id,
        "timezone": timezone,
    }
    if marker is not None:
        settings["marker"] = marker

    payload = json.dumps(settings)
    await prisma_client.db.litellm_config.upsert(
        where={"param_name": _CONFIG_KEY},
        data={
            "create": {"param_name": _CONFIG_KEY, "param_value": payload},
            "update": {"param_value": payload},
        },
    )


async def _get_mavvrik_settings() -> dict:
    """Retrieve and decrypt Mavvrik settings from LiteLLM_Config."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    row = await prisma_client.db.litellm_config.find_first(
        where={"param_name": _CONFIG_KEY}
    )
    if row is None or row.param_value is None:
        return {}

    value = row.param_value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}

    encrypted_key = value.get("api_key", "")
    if encrypted_key:
        try:
            value["api_key"] = decrypt_value_helper(
                encrypted_key, key="mavvrik_api_key"
            )
        except Exception:
            value["api_key"] = encrypted_key

    return value


async def is_mavvrik_setup() -> bool:
    """Return True if Mavvrik settings exist in the database OR are all present as env vars."""
    if all(
        os.getenv(v)
        for v in (
            "MAVVRIK_API_KEY",
            "MAVVRIK_API_ENDPOINT",
            "MAVVRIK_CONNECTION_ID",
        )
    ):
        return True

    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return False

        row = await prisma_client.db.litellm_config.find_first(
            where={"param_name": _CONFIG_KEY}
        )
        return row is not None and row.param_value is not None
    except Exception:
        return False


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


# ------------------------------------------------------------------
# POST /mavvrik/init
# ------------------------------------------------------------------


@router.post(
    "/mavvrik/init",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikInitResponse,
)
async def init_mavvrik_settings(
    request: MavvrikInitRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Initialize Mavvrik settings and store encrypted credentials in the database.

    Calls the Mavvrik register endpoint to obtain the initial metricsMarker (the
    earliest date Mavvrik wants LiteLLM to export from). If that call fails, the
    marker defaults to the first day of the current month.

    After saving settings, the background export job is registered with the
    APScheduler so exports begin automatically without restarting the proxy.
    """
    _require_admin(user_api_key_dict)

    try:
        initial_marker: Optional[str] = None
        try:
            streamer = MavvrikStreamer(
                api_key=request.api_key,
                api_endpoint=request.api_endpoint,
                connection_id=request.connection_id,
            )
            initial_marker = streamer.register()
            verbose_proxy_logger.info(
                "Mavvrik register returned initial marker: %s", initial_marker
            )
        except Exception as reg_exc:
            now = datetime.now(_tz.utc)
            initial_marker = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            verbose_proxy_logger.warning(
                "Mavvrik register call failed (%s) — defaulting marker to first of month: %s",
                reg_exc,
                initial_marker,
            )

        await _set_mavvrik_settings(
            api_key=request.api_key,
            api_endpoint=request.api_endpoint,
            connection_id=request.connection_id,
            timezone=request.timezone,
            marker=initial_marker,
        )
        verbose_proxy_logger.info(
            "Mavvrik settings initialized, marker=%s", initial_marker
        )

        try:
            import litellm.proxy.proxy_server as _pserver

            mavvrik_logger = MavvrikLogger(
                api_key=request.api_key,
                api_endpoint=request.api_endpoint,
                connection_id=request.connection_id,
                timezone=request.timezone,
            )
            litellm.logging_callback_manager.add_litellm_success_callback(mavvrik_logger)
            litellm.logging_callback_manager.add_litellm_async_success_callback(mavvrik_logger)

            if "mavvrik" in litellm.success_callback:
                litellm.success_callback.remove("mavvrik")
            if "mavvrik" in litellm._async_success_callback:
                litellm._async_success_callback.remove("mavvrik")

            _scheduler = getattr(_pserver, "scheduler", None)
            if _scheduler is not None:
                _scheduler.add_job(
                    mavvrik_logger.initialize_mavvrik_export_job,
                    "interval",
                    minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
                    id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
                    replace_existing=True,
                )
                verbose_proxy_logger.info(
                    "Mavvrik background export job scheduled every %d min",
                    MAVVRIK_EXPORT_INTERVAL_MINUTES,
                )
            else:
                verbose_proxy_logger.warning(
                    "Mavvrik: scheduler not available, background job not registered"
                )
        except Exception as sched_exc:
            verbose_proxy_logger.warning(
                "Mavvrik: could not register background job after init (%s)", sched_exc
            )

        return MavvrikInitResponse(
            message="Mavvrik settings initialized successfully", status="success"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ------------------------------------------------------------------
# GET /mavvrik/settings
# ------------------------------------------------------------------


@router.get(
    "/mavvrik/settings",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikSettingsView,
)
async def get_mavvrik_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """View current Mavvrik settings. The API key is masked in the response."""
    _require_admin(user_api_key_dict)

    try:
        settings = await _get_mavvrik_settings()
        if not settings:
            return MavvrikSettingsView(
                api_key_masked=None, marker=None, status="not_configured"
            )

        masked = _sensitive_masker.mask_dict({"api_key": settings.get("api_key", "")})
        return MavvrikSettingsView(
            api_key_masked=masked.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            connection_id=settings.get("connection_id"),
            timezone=settings.get("timezone"),
            marker=settings.get("marker"),
            status="configured",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ------------------------------------------------------------------
# PUT /mavvrik/settings
# ------------------------------------------------------------------


@router.put(
    "/mavvrik/settings",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikInitResponse,
)
async def update_mavvrik_settings(
    request: MavvrikSettingsUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update one or more Mavvrik settings fields. All fields are optional.

    Use the `marker` field to reset the export cursor to a specific date (YYYY-MM-DD),
    for example when Mavvrik resets their metricsMarker and asks you to re-export
    from an earlier date.
    """
    _require_admin(user_api_key_dict)

    if not any(v is not None for v in request.model_dump().values()):
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one field must be provided for update"},
        )

    try:
        current = await _get_mavvrik_settings()

        def _pick(new: Optional[str], key: str, default: str = "") -> str:
            return new if new is not None else current.get(key, default)

        await _set_mavvrik_settings(
            api_key=_pick(request.api_key, "api_key"),
            api_endpoint=_pick(request.api_endpoint, "api_endpoint"),
            connection_id=_pick(request.connection_id, "connection_id"),
            timezone=_pick(request.timezone, "timezone", "UTC"),
            marker=request.marker if request.marker is not None else current.get("marker"),
        )
        return MavvrikInitResponse(
            message="Mavvrik settings updated successfully", status="success"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ------------------------------------------------------------------
# DELETE /mavvrik/delete
# ------------------------------------------------------------------


@router.delete(
    "/mavvrik/delete",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikDeleteResponse,
)
async def delete_mavvrik_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Remove all Mavvrik settings from the database.

    Also deregisters the background export job from the scheduler if it is running.
    """
    _require_admin(user_api_key_dict)

    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        row = await prisma_client.db.litellm_config.find_first(
            where={"param_name": _CONFIG_KEY}
        )
        if row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "Mavvrik is not configured"},
            )

        await prisma_client.db.litellm_config.delete(
            where={"param_name": _CONFIG_KEY}
        )

        # Deregister the scheduler job if present
        try:
            import litellm.proxy.proxy_server as _pserver

            _scheduler = getattr(_pserver, "scheduler", None)
            if _scheduler is not None:
                try:
                    _scheduler.remove_job(MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME)
                except Exception:
                    pass
        except Exception:
            pass

        verbose_proxy_logger.info("Mavvrik settings deleted")
        return MavvrikDeleteResponse(
            message="Mavvrik settings deleted successfully", status="success"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ------------------------------------------------------------------
# POST /mavvrik/dry-run
# ------------------------------------------------------------------


@router.post(
    "/mavvrik/dry-run",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikDryRunResponse,
)
async def dry_run_mavvrik_export(
    request: MavvrikExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Preview the CSV records that would be uploaded for a given date without sending data to GCS.

    Defaults to yesterday if `date_str` is not provided.
    """
    _require_admin(user_api_key_dict)

    try:
        settings = await _get_mavvrik_settings()
        date_str = request.date_str or (
            datetime.now(_tz.utc).date() - timedelta(days=1)
        ).isoformat()

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            connection_id=settings.get("connection_id"),
            timezone=settings.get("timezone", "UTC"),
        )
        result = await logger.dry_run_export_usage_data(
            date_str=date_str, limit=request.limit
        )
        return MavvrikDryRunResponse(
            message="Mavvrik dry run completed",
            status="success",
            dry_run_data={
                "usage_data": result["usage_data"],
                "csv_preview": result["csv_preview"],
            },
            summary=result["summary"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ------------------------------------------------------------------
# POST /mavvrik/export
# ------------------------------------------------------------------


@router.post(
    "/mavvrik/export",
    tags=["Mavvrik"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MavvrikExportResponse,
)
async def export_mavvrik_data(
    request: MavvrikExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Manually trigger a Mavvrik export for a specific date (uploads to GCS via signed URL).

    Defaults to yesterday if `date_str` is not provided. Re-uploading the same date
    overwrites the existing GCS object — exports are idempotent.
    """
    _require_admin(user_api_key_dict)

    try:
        settings = await _get_mavvrik_settings()
        if not settings:
            raise HTTPException(
                status_code=400,
                detail={"error": "Mavvrik not configured. Call POST /mavvrik/init first."},
            )

        date_str = request.date_str or (
            datetime.now(_tz.utc).date() - timedelta(days=1)
        ).isoformat()

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            connection_id=settings.get("connection_id"),
            timezone=settings.get("timezone", "UTC"),
        )
        records_exported = await logger.export_usage_data(
            date_str=date_str,
            limit=request.limit,
        )
        return MavvrikExportResponse(
            message=f"Mavvrik export completed successfully for {date_str}",
            status="success",
            records_exported=records_exported,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
