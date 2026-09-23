"""The `/public/v1` unauthenticated public surface."""

from typing import Final

from fastapi import APIRouter

from litellm.proxy.public_endpoints.public_v1.model_hub import router as model_hub_router

PUBLIC_V1_PREFIX: Final = "/public/v1"

router: Final = APIRouter(prefix=PUBLIC_V1_PREFIX)
router.include_router(model_hub_router)

__all__ = ("PUBLIC_V1_PREFIX", "router")
