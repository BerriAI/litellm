from fastapi.routing import APIRoute

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.debug_utils import router as debug_router
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    router as spend_router,
)


def _get_route_dependency_calls(router, path: str, method: str):
    for route in router.routes:
        if (
            isinstance(route, APIRoute)
            and route.path == path
            and method in route.methods
        ):
            return [dependency.call for dependency in route.dependant.dependencies]
    raise AssertionError(f"Route {method} {path} not found")


def test_sensitive_debug_routes_require_auth_dependency():
    for path, method in (
        ("/debug/asyncio-tasks", "GET"),
        ("/otel-spans", "GET"),
    ):
        assert user_api_key_auth in _get_route_dependency_calls(
            debug_router, path, method
        )


def test_provider_budgets_requires_auth_dependency():
    assert user_api_key_auth in _get_route_dependency_calls(
        spend_router, "/provider/budgets", "GET"
    )
