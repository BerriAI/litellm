from collections.abc import Generator, Iterator
from typing import Final

import pytest

from litellm.responses import dispatch
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.responses.data import RESPONSES_RESPONSE
from tests.test_litellm_rust.responses.support import FallbackProbe
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import native_route


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
    return recording_server


@pytest.fixture(autouse=True)
def native() -> Generator[None]:
    with native_route(Route.RESPONSES):
        yield


@pytest.fixture
def native_responses() -> Iterator[FallbackProbe]:
    probe: Final = FallbackProbe()
    python_aresponses: Final = dispatch._PYTHON_ARESPONSES  # pyright: ignore[reportPrivateUsage]  # the fallback under observation

    async def python_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return await python_aresponses(*args, **kwargs)

    with (
        rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_IN), *catalog.RULES)),
        rebound(dispatch, "_PYTHON_ARESPONSES", python_fallback),
    ):
        yield probe
