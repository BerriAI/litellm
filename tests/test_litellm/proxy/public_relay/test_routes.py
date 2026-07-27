from fastapi.routing import APIRoute

from litellm.proxy.public_relay.admin_endpoints import router as admin_router
from litellm.proxy.public_relay.auth_endpoints import router as auth_router
from litellm.proxy.public_relay.portal_endpoints import router as portal_router
from litellm.proxy.public_relay.public_endpoints import router as public_router


def _paths() -> set[str]:
    return {
        route.path
        for router in (admin_router, auth_router, portal_router, public_router)
        for route in router.routes
        if isinstance(route, APIRoute)
    }


def test_enterprise_routes_are_exposed() -> None:
    paths = _paths()
    assert "/v1/public/auth/activate" in paths
    assert "/v1/admin/relay/accounts" in paths
    assert "/v1/portal/pricing" in paths


def test_consumer_billing_and_registration_routes_are_absent() -> None:
    paths = _paths()
    assert "/v1/public/auth/register" not in paths
    assert "/v1/public/payments/stripe/webhook" not in paths
    assert "/v1/portal/billing/checkout" not in paths
    assert "/v1/admin/relay/payments/{payment_id}/refund" not in paths
