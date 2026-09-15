from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_RESPONSE, call_native_aocr

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


class ReplaceOCRMarkdown(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(
            guardrail_name="replace-ocr-markdown", event_hook=GuardrailEventHooks.post_call, default_on=True
        )
        self.call_types: list[CallTypes] = []

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        self.call_types.append(call_type)
        reviewed_page: Final = response.pages[0].model_copy(update={"markdown": "Reviewed OCR"})
        return response.model_copy(update={"pages": [reviewed_page]})


@pytest.mark.asyncio
async def test_native_aocr_post_call_content_filter_blocks_matching_markdown(
    ocr_server: RecordingServer,
) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="block-native-ocr-markdown",
        event_hook=GuardrailEventHooks.post_call,
        blocked_words=[BlockedWord(keyword="native OCR response", action=ContentFilterAction.BLOCK)],
    )
    litellm.callbacks.append(guardrail)

    with pytest.raises(HTTPException, match="Content blocked") as blocked:
        await call_native_aocr(ocr_server, guardrails=[guardrail.guardrail_name])

    assert blocked.value.status_code == 400
    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
async def test_native_aocr_post_call_replacement_reaches_caller_and_success_callback(
    ocr_server: RecordingServer,
) -> None:
    guardrail: Final = ReplaceOCRMarkdown()
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await call_native_aocr(
        ocr_server,
        callbacks=[recorder],
        guardrails=[guardrail.guardrail_name],
    )
    success_events: Final = await recorder.wait_for_async("async_log_success_event")

    assert guardrail.call_types == [CallTypes.aocr]
    assert response.pages[0].markdown == "Reviewed OCR"
    assert len(success_events) == 1
    assert success_events[0].response.pages[0].markdown == "Reviewed OCR"
    assert "guardrails" not in ocr_server.requests[0].body
