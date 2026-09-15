from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_completion_extensions_cross_python_rust_and_http(
    recording_server: RecordingServer, asynchronous: bool
) -> None:
    recording_server.default_response = ResponseSpec(
        body={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
    )
    options: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": recording_server.base_url,
        "api_key": "sk-test",
        "max_tokens": 16,
        "future_provider_option": {"nested": [None, False, 0]},
        "extra_body": {"another_future_option": {"enabled": True}},
    }
    response: Final = await litellm.acompletion(**options) if asynchronous else litellm.completion(**options)

    assert response.choices[0].message.content == "hello"
    assert len(recording_server.requests) == 1
    request: Final = recording_server.requests[0]
    assert not request.headers.get("user-agent", "").startswith("python-httpx")
    assert request.body["future_provider_option"] == {"nested": [None, False, 0]}
    assert request.body["another_future_option"] == {"enabled": True}
    assert request.body["max_tokens"] == 16
    assert "extra_body" not in request.body
    assert "api_key" not in request.body
