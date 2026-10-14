from typing import Final

import pytest

from litellm.messages.dispatch import ANTHROPIC_MESSAGES
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, RESPONSES_RESPONSE, RESPONSES_TEXT
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import python_only, rust_required
from tests.test_litellm_rust.support.served import assert_nested_call_served_natively

pytestmark = pytest.mark.requires_rust_extension

LARGE_PROMPT: Final = "x" * 300_000


async def _messages_over_native_responses(server: RecordingServer, prompt: str) -> tuple[object, RecordingLogger]:
    server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
    recorder: Final = RecordingLogger()
    result: Final = await ANTHROPIC_MESSAGES.arun(
        (),
        {
            "model": RESPONSES_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "api_key": "test-key",
            "api_base": server.base_url,
            "callbacks": [recorder],
        },
        rules=(*python_only(Route.MESSAGES), *rust_required(Route.RESPONSES)),
    )
    await drain_logging()
    return result, recorder


@pytest.mark.asyncio
async def test_nested_responses_call_follows_the_outer_call_rules(recording_server: RecordingServer) -> None:
    result, recorder = await _messages_over_native_responses(recording_server, "Hello")

    assert_nested_call_served_natively(recording_server, recorder)
    assert isinstance(result, dict)
    assert result["content"][0]["text"] == RESPONSES_TEXT


@pytest.mark.asyncio
@gap("the native host never claims the async success log, so the outer Python wrapper logs the request again")
async def test_messages_over_native_responses_logs_a_large_request_once(recording_server: RecordingServer) -> None:
    recorder: Final = (await _messages_over_native_responses(recording_server, LARGE_PROMPT))[1]

    assert recorder.names.count("async_log_success_event") == 1
    assert_nested_call_served_natively(recording_server, recorder)
