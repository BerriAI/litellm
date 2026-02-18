"""FastAPI admin endpoints for the Mavvrik integration.

Endpoints (all require PROXY_ADMIN role):
    POST /mavvrik/init          Store encrypted settings in LiteLLM_Config
    GET  /mavvrik/settings      View current settings (API key masked)
    PUT  /mavvrik/settings      Update existing settings
    POST /mavvrik/dry-run       Preview NDJSON records without uploading
    POST /mavvrik/export        Trigger manual upload to GCS
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.proxy.mavvrik_endpoints import (
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


# ------------------------------------------------------------------
# Internal settings helpers
# ------------------------------------------------------------------


async def _set_mavvrik_settings(
    api_key: str,
    api_endpoint: str,
    tenant: str,
    instance_id: str,
    timezone: str,
    marker: Optional[str] = None,
):
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
        "tenant": tenant,
        "instance_id": instance_id,
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

    # Decrypt API key
    encrypted_key = value.get("api_key", "")
    if encrypted_key:
        try:
            value["api_key"] = decrypt_value_helper(
                encrypted_key, key="mavvrik_api_key"
            )
        except Exception:
            value["api_key"] = encrypted_key  # fall back to raw value

    return value


async def is_mavvrik_setup() -> bool:
    """Return True if Mavvrik settings exist in the database."""
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


def _mask_key(api_key: str) -> str:
    if not api_key or len(api_key) < 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"


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
    """Initialize Mavvrik settings and store encrypted API key in the database."""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Call Mavvrik register endpoint to get the initial metricsMarker
        # (the epoch timestamp Mavvrik wants LiteLLM to start from).
        # This is best-effort: if the endpoint is unavailable or the format
        # changes, we fall back to first-of-month and still save the settings.
        initial_marker: Optional[str] = None
        try:
            from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer

            streamer = MavvrikStreamer(
                api_key=request.api_key,
                api_endpoint=request.api_endpoint,
                tenant=request.tenant,
                instance_id=request.instance_id,
            )
            initial_marker = streamer.register()
            verbose_proxy_logger.info(
                "Mavvrik register returned initial marker: %s", initial_marker
            )
        except Exception as reg_exc:
            verbose_proxy_logger.warning(
                "Mavvrik register call failed (%s) — using first-of-month default as marker",
                reg_exc,
            )

        await _set_mavvrik_settings(
            api_key=request.api_key,
            api_endpoint=request.api_endpoint,
            tenant=request.tenant,
            instance_id=request.instance_id,
            timezone=request.timezone,
            marker=initial_marker,
        )
        verbose_proxy_logger.info(
            "Mavvrik settings initialized, marker=%s", initial_marker
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
    """View current Mavvrik settings (API key is masked)."""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        settings = await _get_mavvrik_settings()
        if not settings:
            return MavvrikSettingsView(
                api_key_masked=None, marker=None, status="not_configured"
            )

        return MavvrikSettingsView(
            api_key_masked=_mask_key(settings.get("api_key", "")),
            api_endpoint=settings.get("api_endpoint"),
            tenant=settings.get("tenant"),
            instance_id=settings.get("instance_id"),
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
    """Update one or more Mavvrik settings fields."""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    if not any(v is not None for v in request.model_dump().values()):
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one field must be provided for update"},
        )

    try:
        current = await _get_mavvrik_settings()

        def _pick(new, key, default=""):
            return new if new is not None else current.get(key, default)

        await _set_mavvrik_settings(
            api_key=_pick(request.api_key, "api_key"),
            api_endpoint=_pick(request.api_endpoint, "api_endpoint"),
            tenant=_pick(request.tenant, "tenant"),
            instance_id=_pick(request.instance_id, "instance_id"),
            timezone=_pick(request.timezone, "timezone", "UTC"),
            marker=current.get("marker"),
        )
        return MavvrikInitResponse(
            message="Mavvrik settings updated successfully", status="success"
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
    """Preview NDJSON records that would be uploaded — no data is sent to GCS."""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        settings = await _get_mavvrik_settings()
        from litellm.integrations.mavvrik.mavvrik import MavvrikLogger

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            tenant=settings.get("tenant"),
            instance_id=settings.get("instance_id"),
            timezone=settings.get("timezone", "UTC"),
        )
        result = await logger.dry_run_export_usage_data(limit=request.limit)
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
    """Manually trigger a Mavvrik export (upload to GCS via signed URL)."""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    from datetime import datetime, timedelta, timezone as _tz

    try:
        settings = await _get_mavvrik_settings()
        if not settings:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Mavvrik not configured. Call POST /mavvrik/init first."
                },
            )

        # Default to yesterday when no date provided
        date_str = request.date_str or (
            datetime.now(_tz.utc).date() - timedelta(days=1)
        ).isoformat()

        from litellm.integrations.mavvrik.mavvrik import MavvrikLogger

        logger = MavvrikLogger(
            api_key=settings.get("api_key"),
            api_endpoint=settings.get("api_endpoint"),
            tenant=settings.get("tenant"),
            instance_id=settings.get("instance_id"),
            timezone=settings.get("timezone", "UTC"),
        )
        await logger.export_usage_data(
            date_str=date_str,
            limit=request.limit,
        )
        return MavvrikExportResponse(
            message=f"Mavvrik export completed successfully for {date_str}",
            status="success",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
