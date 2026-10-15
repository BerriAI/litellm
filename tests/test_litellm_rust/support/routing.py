import asyncio
import copy
from collections.abc import Awaitable, Mapping
from typing import Final, Protocol

from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.served import assert_served_natively

__all__ = ["CALL_ID", "assert_served_natively", "call_natively", "native_route"]

CALL_ID: Final = "callback-contract"


class Creates(Protocol):
    def __call__(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> object: ...


class Acreates(Protocol):
    def __call__(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> Awaitable[object]: ...


def rust_required(route: Route) -> Rules:
    return (RouteRule(route, Rollout.RUST_REQUIRED),)


def python_only(route: Route) -> Rules:
    return (RouteRule(route, Rollout.PYTHON_ONLY),)


async def call_natively(
    route: Route,
    create: Creates,
    acreate: Acreates,
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
    rules: Final = rust_required(route)
    if asynchronous:
        return await acreate((), arguments, rules=rules)
    return await asyncio.to_thread(create, (), arguments, rules=rules)


