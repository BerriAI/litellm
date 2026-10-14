from collections.abc import Generator
from contextlib import contextmanager

from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer


@contextmanager
def native_route(*routes: Route) -> Generator[None]:
    rules: tuple[RouteRule, ...] = tuple(RouteRule(route, Rollout.RUST_REQUIRED) for route in routes)
    with rebound(catalog, "RULES", (*rules, *catalog.RULES)):
        yield


def assert_served_natively(server: RecordingServer, count: int = 1) -> None:
    assert len(server.requests) == count
    assert not any(request.headers.get("user-agent", "").startswith("python-httpx") for request in server.requests)
