import asyncio
import copy
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Final

from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.served import assert_served_natively

__all__ = ["CALL_ID", "assert_served_natively", "call_public", "native_route"]

CALL_ID: Final = "callback-contract"


@contextmanager
def native_route(*routes: Route) -> Generator[None]:
    with rebound(catalog, "RULES", (*(RouteRule(route, Rollout.RUST_REQUIRED) for route in routes), *catalog.RULES)):
        yield


async def call_public(
    create: Callable[..., object],
    acreate: Callable[..., Awaitable[object]],
    request: Mapping[str, object],
    server: RecordingServer,
    *,
    asynchronous: bool,
    **kwargs: object,
) -> object:
    arguments: Final = {
        **copy.deepcopy(dict(request)),
        "api_key": "test-key",
        "api_base": server.base_url,
        "litellm_call_id": CALL_ID,
        **kwargs,
    }
    if asynchronous:
        return await acreate(**arguments)
    return await asyncio.to_thread(create, **arguments)
