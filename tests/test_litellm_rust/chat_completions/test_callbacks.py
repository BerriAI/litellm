from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes, ModelResponse
from tests.test_litellm_rust.chat_completions.data import CHAT_SURFACE
from tests.test_litellm_rust.messages.data import MESSAGES_TEXT
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.requests import request_body
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_returns_a_model_response_and_logs_success_once(server: RecordingServer, asynchronous: bool) -> None:
    recorder: Final = RecordingLogger()

    result: Final = await CHAT_SURFACE.call(server, asynchronous=asynchronous, callbacks=[recorder], temperature=0.25)

    assert_served_natively(server)
    sent: Final = server.requests[0]
    body: Final = _OBJECT.validate_python(sent.body)
    assert sent.path == "/v1/messages"
    assert body["temperature"] == 0.25
    assert request_body(_OBJECT.validate_python(recorder.wait_for("log_pre_api_call")[0].kwargs)) == body
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == MESSAGES_TEXT
    success: Final = await recorder.wait_for_async("async_log_success_event" if asynchronous else "log_success_event")
    assert len(success) == 1
    assert success[0].response is result


@pytest.mark.asyncio
async def test_pre_api_call_edits_reach_the_provider(server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.75

    await CHAT_SURFACE.call(server, callbacks=[Edit()])

    assert _OBJECT.validate_python(server.requests[0].body)["temperature"] == 0.75


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_can_delete_keywords(server: RecordingServer) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("temperature", "extra_headers")}

    litellm.callbacks.append(DropKeywords())
    await CHAT_SURFACE.call(server, temperature=0.25, extra_headers={"x-deleted": "remove this header"})

    assert_served_natively(server)
    sent: Final = server.requests[0]
    assert "temperature" not in _OBJECT.validate_python(sent.body)
    assert sent.headers.get("x-deleted") is None
