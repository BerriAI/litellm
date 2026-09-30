"""Route table contract for the LITELLM_DISABLE_LAZY_ROUTES startup flag.

By default optional feature routers (``LAZY_FEATURES``) are registered on the first
request to their path prefix, so an operator inspecting the route table right after
boot cannot see or gate them. With the flag set every feature is registered at worker
startup, so ``GET /routes`` lists them before any feature request is served and the
first feature request changes nothing.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy._lazy_features import LAZY_FEATURES
from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.process import owned_proxy_process

TICKET_FEATURES: Final = ("mcp_management", "mcp_byok_oauth")


def _routed_features(candidate: Gateway) -> Mapping[str, tuple[str, ...]]:
    routes: Final = candidate.get("/routes")["routes"]
    assert isinstance(routes, list), routes
    paths: Final = tuple(string_value(object_value(route)["path"]) for route in routes)
    return {feature.name: tuple(path for path in paths if feature.matches(path)) for feature in LAZY_FEATURES}


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
