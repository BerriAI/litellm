import asyncio
from typing import Final

import pytest

import litellm
from litellm.chat_completions.dispatch import ACOMPLETION
from litellm.exceptions import APIConnectionError
from litellm.rust_bridge import _native
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.chat_completions.data import REQUEST
from tests.test_litellm_rust.messages.data import MESSAGES
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, RESPONSES_RESPONSE, RESPONSES_TEXT
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import python_only, rust_required
from tests.test_litellm_rust.support.served import assert_nested_call_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_unstarted_call_has_no_provider_or_callback_effects(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = RecordingLogger()
    server.expected_requests = 0
    monkeypatch.setattr(litellm, "ssl_verify", object())
    monkeypatch.setattr(litellm, "max_budget", 1.0)
    monkeypatch.setattr(litellm, "_current_cost", 2.0)
    kwargs: Final = {
        **REQUEST,
        "api_key": "test-key",
        "api_base": server.base_url,
        "callbacks": [recorder],
    }

    pending: Final = _native.acompletion(NativeCall(args=(), kwargs=kwargs, base={}))
    assert asyncio.iscoroutine(pending)
    pending.close()

    assert not server.requests
    assert not recorder.events


@pytest.mark.asyncio
@gap(
    "the chat to Responses bridge forwards chat-only keys such as acompletion, so native Responses declines it",
    raises=APIConnectionError,
)
async def test_chat_bridged_to_native_responses_logs_the_request_once(recording_server: RecordingServer) -> None:
    recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
    recording_server.expected_requests = None
    recorder: Final = RecordingLogger()

    result: Final = await ACOMPLETION.arun(
        (),
        {
            "model": f"openai/responses/{RESPONSES_MODEL.removeprefix('openai/')}",
            "messages": list(MESSAGES),
            "api_key": "test-key",
            "api_base": recording_server.base_url,
            "callbacks": [recorder],
        },
        rules=(*python_only(Route.CHAT_COMPLETIONS), *rust_required(Route.RESPONSES)),
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == RESPONSES_TEXT
    await drain_logging()
    assert_nested_call_served_natively(recording_server, recorder)
    assert recorder.names.count("async_log_success_event") == 1
