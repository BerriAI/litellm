"""
Characterization gap: a streaming adapter completion splits its timeouts like a streaming chat
completion, sending the request timeout as ``timeout`` and the stream timeout beside it, so a
fake-streaming provider stays bounded by the request timeout rather than the stream timeout.

Every test builds its core through ``make_core`` on the ``host`` fixture, so it runs once per core.
"""

from collections.abc import Callable
from typing import Final

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.scripted_host import deployment

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]


@pytest.fixture
def host() -> EndpointHost:
    return EndpointHost()


@pytest.mark.parametrize(("stream", "expected"), [(False, (11.0, None)), (True, (11.0, 5.0))])
async def test_adapter_completion_sends_the_request_timeout_and_the_stream_timeout_beside_it(
    make_core: MakeCore, host: EndpointHost, stream: bool, expected: tuple[float, float | None]
):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11)], stream_timeout=5)
    await core.aadapter_completion(adapter_id="my-adapter", model="g", messages=MESSAGES, stream=stream)
    sent: Final = host.calls[-1].kwargs
    assert (sent["timeout"], sent.get("stream_timeout")) == expected
