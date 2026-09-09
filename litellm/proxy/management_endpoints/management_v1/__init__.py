"""The `/management/v1` control-plane surface."""

from typing import Final

from fastapi import APIRouter

from litellm.proxy.management_endpoints.management_v1.budgets import (
    router as budgets_router,
)
from litellm.proxy.management_endpoints.management_v1.spend_logs import (
    router as spend_logs_router,
)

router: Final = APIRouter()
router.include_router(budgets_router)
router.include_router(spend_logs_router)

__all__ = ["router"]
