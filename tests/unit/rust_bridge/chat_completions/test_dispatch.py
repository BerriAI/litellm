import asyncio
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

from litellm.chat_completions.dispatch import ACOMPLETION, COMPLETION
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE, MESSAGES_TEXT
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.served import assert_served_natively

pytest.importorskip("litellm.rust_bridge._native")

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

RUST: Final = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)
CALLS: Final = pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((MESSAGES_MODEL, list(MESSAGES)), {"max_tokens": 16}, id="positional"),
        pytest.param((), {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 16}, id="keyword"),
    ),
)


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with recording_service() as recording:
        recording.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        yield recording


def assert_served_natively_with_text(server: RecordingServer, response: object) -> None:
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == MESSAGES_TEXT
    assert_served_natively(server, response)


@CALLS
def test_completion_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    response: Final = COMPLETION.run(args, {**kwargs, "api_key": "test-key", "api_base": server.base_url}, rules=RUST)
    assert_served_natively_with_text(server, response)


@pytest.mark.asyncio
@CALLS
async def test_acompletion_reaches_native_from_positional_and_keyword_calls(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> None:
    response: Final = await ACOMPLETION.arun(
        args, {**kwargs, "api_key": "test-key", "api_base": server.base_url}, rules=RUST
    )
    assert_served_natively_with_text(server, response)


@pytest.mark.asyncio
async def test_completion_with_the_acompletion_flag_binds_positional_arguments_by_the_sync_signature(
    server: RecordingServer,
) -> None:
    pending: Final = COMPLETION.run(
        (MESSAGES_MODEL, list(MESSAGES), 30, 0.25),
        {"max_tokens": 16, "api_key": "test-key", "api_base": server.base_url, "acompletion": True},
        rules=RUST,
    )

    assert asyncio.iscoroutine(pending)
    assert_served_natively_with_text(server, await pending)
    body: Final = server.requests[0].body
    assert isinstance(body, dict)
    assert body["temperature"] == 0.25
    assert "functions" not in body
