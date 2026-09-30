"""Route table contract for the LITELLM_DISABLE_LAZY_ROUTES startup flag.

By default optional feature routers (``LAZY_FEATURES``) are registered on the first
request to their path prefix, so an operator inspecting the route table right after
boot cannot see or gate them. With the flag set every feature is registered at worker
startup, so ``GET /routes`` lists them before any feature request is served and the
first feature request changes nothing.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy._lazy_features import LAZY_FEATURES
from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.process import owned_proxy_process

TICKET_FEATURES: Final = ("mcp_management", "mcp_byok_oauth")
HOOK_MODULE: Final = "lazy_routes_route_filter_hook"
HOOK_SOURCE: Final = """from litellm.proxy.proxy_server import app


def drop_mcp_routes() -> None:
    app.router.routes[:] = [
        route for route in app.router.routes if not getattr(route, "path", "").startswith(("/mcp", "/v1/mcp"))
    ]
"""


def _paths(candidate: Gateway) -> tuple[str, ...]:
    routes: Final = candidate.get("/routes")["routes"]
    assert isinstance(routes, list), routes
    return tuple(string_value(object_value(route)["path"]) for route in routes)


def _routed_features(candidate: Gateway) -> Mapping[str, tuple[str, ...]]:
    paths: Final = _paths(candidate)
    return {feature.name: tuple(path for path in paths if feature.matches(path)) for feature in LAZY_FEATURES}


def _mcp_paths(candidate: Gateway) -> tuple[str, ...]:
    return tuple(path for path in _paths(candidate) if path.startswith(("/mcp", "/v1/mcp")))


def _route_filter_hook(directory: Path) -> Mapping[str, str]:
    (directory / f"{HOOK_MODULE}.py").write_text(HOOK_SOURCE)
    search_path: Final = (str(directory), os.environ.get("PYTHONPATH", ""))
    return {
        "PYTHONPATH": os.pathsep.join(entry for entry in search_path if entry),
        "LITELLM_WORKER_STARTUP_HOOKS": f"{HOOK_MODULE}:drop_mcp_routes",
    }


def test_lazy_routes_are_absent_from_the_route_table_until_first_request(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        assert {name: at_boot[name] for name in TICKET_FEATURES} == {name: () for name in TICKET_FEATURES}, at_boot
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        after_first_request: Final = _routed_features(owned.gateway)
        assert after_first_request["mcp_management"] != (), "first request did not register the router"
        assert after_first_request["mcp_byok_oauth"] == (), "only the requested feature is mounted"


@pytest.mark.parametrize("workers", (1, 4))
def test_disable_lazy_routes_flag_registers_every_feature_at_startup(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    with owned_proxy_process(gateway, tmp_path, {"LITELLM_DISABLE_LAZY_ROUTES": "true"}, workers=workers) as owned:
        at_boot: Final = tuple(_routed_features(owned.gateway) for _ in range(2 * workers))
        unregistered: Final = sorted(name for name, paths in at_boot[0].items() if not paths)
        assert unregistered == [], f"features still missing from /routes at startup: {unregistered}"
        assert all(table == at_boot[0] for table in at_boot), "workers disagree on the route table"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _routed_features(owned.gateway) == at_boot[0], "first feature request changed the route table"


def test_startup_hook_cannot_remove_lazy_routes_that_register_after_it_ran(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, _route_filter_hook(tmp_path)) as owned:
        assert _mcp_paths(owned.gateway) == (), "hook should have removed the routes registered before it ran"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _mcp_paths(owned.gateway) != (), "first request should have registered the routes the hook never saw"


def test_disable_lazy_routes_flag_lets_a_startup_hook_remove_optional_routes_for_good(
    gateway: Gateway, tmp_path: Path
) -> None:
    overrides: Final = {**_route_filter_hook(tmp_path), "LITELLM_DISABLE_LAZY_ROUTES": "true"}
    with owned_proxy_process(gateway, tmp_path, overrides) as owned:
        assert _mcp_paths(owned.gateway) == (), "hook should have seen and removed every MCP route"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 404, listing.text
        mounted: Final = owned.gateway.request("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert mounted.status_code == 404, mounted.text
        guardrails: Final = owned.gateway.request("GET", "/guardrails/list")
        assert guardrails.status_code == 200, guardrails.text
        assert _mcp_paths(owned.gateway) == (), "a feature request re-registered routes the hook removed"
