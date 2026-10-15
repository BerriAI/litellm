import asyncio
import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.responses import main as python_responses
from litellm.responses.dispatch import ARESPONSES, RESPONSES
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, RESPONSES_RESPONSE, RESPONSES_TEXT
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.served import assert_served_natively

pytest.importorskip("litellm.rust_bridge._native")

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

RUST: Final = (RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),)
CALLS: Final = pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param(("Hello", RESPONSES_MODEL, None, "Be brief", 16), {}, id="positional"),
        pytest.param((), {"input": "Hello", "model": RESPONSES_MODEL}, id="keyword"),
    ),
)


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with recording_service() as recording:
        recording.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        yield recording


def assert_served_natively_with_text(server: RecordingServer, response: object) -> None:
    assert isinstance(response, ResponsesAPIResponse)
    assert response.output_text == RESPONSES_TEXT
    assert_served_natively(server, response)


@CALLS
def test_responses_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    response: Final = RESPONSES.run(args, {**kwargs, "api_key": "test-key", "api_base": server.base_url}, rules=RUST)
    assert_served_natively_with_text(server, response)


@pytest.mark.asyncio
@CALLS
async def test_aresponses_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    response: Final = await ARESPONSES.arun(
        args, {**kwargs, "api_key": "test-key", "api_base": server.base_url}, rules=RUST
    )
    assert_served_natively_with_text(server, response)


@pytest.mark.asyncio
async def test_responses_with_the_aresponses_flag_returns_a_coroutine_native_serves(server: RecordingServer) -> None:
    pending: Final = RESPONSES.run(
        (),
        {
            "input": "Hello",
            "model": RESPONSES_MODEL,
            "api_key": "test-key",
            "api_base": server.base_url,
            "aresponses": True,
        },
        rules=RUST,
    )

    assert asyncio.iscoroutine(pending)
    assert_served_natively_with_text(server, await pending)


def test_public_signature_is_the_legacy_signature() -> None:
    public_responses: Final = cast(Callable[..., object], litellm.responses)
    legacy_responses: Final = cast(Callable[..., object], python_responses.responses)
    public_aresponses: Final = cast(Callable[..., object], litellm.aresponses)
    legacy_aresponses: Final = cast(Callable[..., object], python_responses.aresponses)
    assert inspect.signature(public_responses) == inspect.signature(legacy_responses)
    assert inspect.signature(public_aresponses) == inspect.signature(legacy_aresponses)
