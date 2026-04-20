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

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException

from litellm.integrations.mavvrik import MavvrikService
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


@asynccontextmanager
async def _mavvrik_errors() -> AsyncIterator[None]:
    """Centralised exception → HTTPException mapping for all Mavvrik endpoints.

    MavvrikService raises typed exceptions that map directly to HTTP status codes:
        LookupError   → 404  (resource not found, e.g. settings not configured)
        ValueError    → 400  (bad input, e.g. missing required field)
        RuntimeError  → 500  (upstream / integration failure)
        Exception     → 500  (unexpected catch-all)

    HTTPException is re-raised as-is (e.g. 403 from _require_admin).
    """
    try:
        yield
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    except (RuntimeError, Exception) as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ---------------------------------------------------------------------------
# POST /mavvrik/init
# ---------------------------------------------------------------------------


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
    async with _mavvrik_errors():
        result = await MavvrikService().initialize(
            api_key=request.api_key,
            api_endpoint=request.api_endpoint,
            connection_id=request.connection_id,
        )
        return MavvrikInitResponse(**result)


# ---------------------------------------------------------------------------
# GET /mavvrik/settings
# ---------------------------------------------------------------------------


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
    async with _mavvrik_errors():
        result = await MavvrikService().get_settings()
        return MavvrikSettingsView(**result)


# ---------------------------------------------------------------------------
# PUT /mavvrik/settings
# ---------------------------------------------------------------------------


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

    The export marker is owned by the Mavvrik API and cannot be set here.
    """
    _require_admin(user_api_key_dict)

    if not any(v is not None for v in request.model_dump().values()):
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one field must be provided for update"},
        )

    async with _mavvrik_errors():
        result = await MavvrikService().update_settings(
            api_key=request.api_key,
            api_endpoint=request.api_endpoint,
            connection_id=request.connection_id,
        )
        return MavvrikInitResponse(**result)


# ---------------------------------------------------------------------------
# DELETE /mavvrik/delete
# ---------------------------------------------------------------------------


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
    async with _mavvrik_errors():
        result = await MavvrikService().delete()
        return MavvrikDeleteResponse(**result)


# ---------------------------------------------------------------------------
# POST /mavvrik/dry-run
# ---------------------------------------------------------------------------


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
    async with _mavvrik_errors():
        result = await MavvrikService().dry_run(
            date_str=request.date_str,
            limit=request.limit,
        )
        return MavvrikDryRunResponse(**result)


# ---------------------------------------------------------------------------
# POST /mavvrik/export
# ---------------------------------------------------------------------------


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
    async with _mavvrik_errors():
        result = await MavvrikService().export(
            date_str=request.date_str,
            limit=request.limit,
        )
        return MavvrikExportResponse(**result)
