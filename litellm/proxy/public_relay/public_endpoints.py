from __future__ import annotations

from fastapi import APIRouter

from litellm.proxy.public_relay.api_types import StatusResponse
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.repository import database_handle
from litellm.proxy.public_relay.runtime import database

router = APIRouter(tags=["public relay"])


@router.get("/v1/public/status", response_model=StatusResponse)
async def relay_status() -> StatusResponse:
    value = PublicRelaySettings.from_env()
    operational = value.enabled and not value.missing_runtime_configuration()
    if operational:
        try:
            await database_handle(database()).query_raw("SELECT 1")
        except Exception:
            operational = False
    return StatusResponse(enabled=value.enabled, operational=operational)
