from pathlib import Path
from typing import Final

from backend.routes.allowlist import BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES
from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES
from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration._support.process import owned_proxy_process

LAZY_ROUTES_FLAG: Final = "LITELLM_DISABLE_LAZY_ROUTES"


def _served_paths(candidate: Gateway) -> frozenset[str]:
    routes: Final = tuple(object_value(route) for route in list_value(candidate.get("/routes")["routes"]))
    return frozenset(string_value(route["path"]) for route in routes if route.get("mounted_app") is not True)


def _kept_by_a_component(path: str) -> bool:
    return (
        path in GATEWAY_EXACT_PATHS
        or path.startswith(GATEWAY_PATH_PREFIXES)
        or path in BACKEND_EXACT_PATHS
        or path.startswith(BACKEND_PATH_PREFIXES)
    )


def test_every_route_a_fresh_proxy_serves_is_kept_by_the_gateway_or_the_backend(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, remove_environment=(LAZY_ROUTES_FLAG,)) as owned:
        served: Final = _served_paths(owned.gateway)
    assert len(served) > 100, sorted(served)
    assert sorted(path for path in served if not _kept_by_a_component(path)) == []
