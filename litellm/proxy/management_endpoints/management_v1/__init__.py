"""The `/management/v1` control-plane surface."""

from fastapi import APIRouter

from litellm.proxy.management_endpoints.management_v1.spend_logs import (
    router as spend_logs_router,
)

router = APIRouter()
router.include_router(spend_logs_router)

__all__ = ["router"]
