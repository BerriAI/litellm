"""The `/management/v1` control-plane surface."""

from typing import Final

from fastapi import APIRouter

from litellm.proxy.management.users.endpoints import router as current_user_router
from litellm.proxy.management_endpoints.management_v1.budgets import (
    router as budgets_router,
)
from litellm.proxy.management_endpoints.management_v1.customer_budget_increase_requests import (
    router as customer_budget_increase_requests_router,
)
from litellm.proxy.management_endpoints.management_v1.customer_usage import (
    router as customer_usage_router,
)
from litellm.proxy.management_endpoints.management_v1.customers import (
    router as customers_router,
)
from litellm.proxy.management_endpoints.management_v1.spend_logs import (
    router as spend_logs_router,
)
from litellm.proxy.management_endpoints.management_v1.teams import (
    router as teams_router,
)
from litellm.proxy.management_endpoints.management_v1.users import (
    router as users_router,
)

router: Final = APIRouter()
router.include_router(budgets_router)
router.include_router(customers_router)
router.include_router(customer_budget_increase_requests_router)
router.include_router(customer_usage_router)
router.include_router(spend_logs_router)
router.include_router(teams_router)
router.include_router(users_router)
router.include_router(current_user_router)

__all__ = ["router"]
