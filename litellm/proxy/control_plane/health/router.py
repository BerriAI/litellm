from typing import Final

from fastapi import APIRouter

from litellm.types.proxy.control_plane.health import HealthResponse

router: Final = APIRouter(tags=["control plane: health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    return HealthResponse(status="ok")
