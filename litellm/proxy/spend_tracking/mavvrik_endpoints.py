"""FastAPI admin endpoints for the Mavvrik integration.

Endpoints (all require PROXY_ADMIN role):
    POST   /mavvrik/init          Store encrypted settings + start background job
    GET    /mavvrik/settings      View current settings (API key masked)
    PUT    /mavvrik/settings      Update existing settings
    DELETE /mavvrik/delete        Remove all Mavvrik settings
    POST   /mavvrik/dry-run       Preview CSV records without uploading
    POST   /mavvrik/export        Trigger a manual upload to Mavvrik

All business logic (scheduling, logger creation, setup detection) lives in
litellm/integrations/mavvrik/ — these handlers are thin dispatchers only.
"""

import os
from datetime import datetime, timedelta
from datetime import timezone as _tz
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.mavvrik.database import LiteLLMDatabase
from litellm.integrations.mavvrik.logger import MavvrikLogger
from litellm.integrations.mavvrik.register import is_mavvrik_setup  # noqa: F401
from litellm.integrations.mavvrik.upload import MavvrikUploader
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

_sensitive_masker = SensitiveDataMasker()


# ------------------------------------------------------------------
# Helpers shared by all handlers
# ------------------------------------------------------------------


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


async def _get_mavvrik_settings() -> dict:
    """Retrieve and decrypt Mavvrik settings from LiteLLM_Config."""
    try:
        db = LiteLLMDatabase()
        value = await db.get_mavvrik_settings()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        ) from exc

    if not value:
        return {}

    encrypted_key = value.get("api_key", "")
    if encrypted_key:
        decrypted = decrypt_value_helper(encrypted_key, key="mavvrik_api_key")
        if decrypted is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to decrypt Mavvrik API key. Check your salt key configuration."
                },
            )
        value["api_key"] = decrypted

    return value


async def _set_mavvrik_settings(
    api_key: str,
    api_endpoint: str,
    connection_id: str,
    marker: Optional[str] = None,
) -> None:
    """Encrypt API key and persist all settings via LiteLLMDatabase."""
    encrypted_api_key = encrypt_value_helper(api_key)
    settings: dict = {
        "api_key": encrypted_api_key,
        "api_endpoint": api_endpoint,
        "connection_id": connection_id,
    }
    if marker is not None:
        settings["marker"] = marker

    try:
        db = LiteLLMDatabase()
        await db.set_mavvrik_settings(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to save Mavvrik settings: {exc}"},
        ) from exc


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
    """Initialize Mavvrik settings and register the background export job."""
    _require_admin(user_api_key_dict)

    try:
        # Get initial marker from Mavvrik (best-effort — fallback to first of month)
        initial_marker: Optional[str] = None
        try:
            uploader = MavvrikUploader(
                api_key=request.api_key,
                api_endpoint=request.api_endpoint,
                connection_id=request.connection_id,
            )
            initial_marker = await uploader.register()
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
            marker=initial_marker,
        )

        # Delegate logger creation + job scheduling to the mavvrik module
        try:
            import litellm.proxy.proxy_server as _pserver

            from litellm.integrations.mavvrik.register import register_logger_and_job

            _scheduler = getattr(_pserver, "scheduler", None)
            await register_logger_and_job(
                api_key=request.api_key,
                api_endpoint=request.api_endpoint,
                connection_id=request.connection_id,
                scheduler=_scheduler,
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

    Use the ``marker`` field to reset the export cursor to a specific date (YYYY-MM-DD)
    when Mavvrik asks you to re-export from an earlier date.
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
            marker=(
                request.marker if request.marker is not None else current.get("marker")
            ),
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
    """Remove all Mavvrik settings and deregister the background job."""
    _require_admin(user_api_key_dict)

    try:
        from litellm.constants import MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        row = await prisma_client.db.litellm_config.find_first(
            where={"param_name": "mavvrik_settings"}
        )
        if row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "Mavvrik is not configured"},
            )

        await prisma_client.db.litellm_config.delete(
            where={"param_name": "mavvrik_settings"}
        )

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
    """Preview the CSV records that would be uploaded for a given date without sending data."""
    _require_admin(user_api_key_dict)

    try:
        settings = await _get_mavvrik_settings()
        date_str = (
            request.date_str
            or (datetime.now(_tz.utc).date() - timedelta(days=1)).isoformat()
        )

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            connection_id=settings.get("connection_id"),
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
    """Manually trigger a Mavvrik export for a specific date."""
    _require_admin(user_api_key_dict)

    try:
        settings = await _get_mavvrik_settings()
        if not settings:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Mavvrik not configured. Call POST /mavvrik/init first."
                },
            )

        date_str = (
            request.date_str
            or (datetime.now(_tz.utc).date() - timedelta(days=1)).isoformat()
        )

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            connection_id=settings.get("connection_id"),
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
