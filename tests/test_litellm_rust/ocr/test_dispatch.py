from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.mark.parametrize("rust_enabled", [True, False], ids=["enabled", "disabled"])
def test_public_ocr_dispatches_according_to_rust_setting(
    ocr_server: RecordingServer,
    rust_enabled: bool,
) -> None:
    litellm.rust(rust_enabled)

    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "native OCR response"
    assert len(ocr_server.requests) == 1
    assert ocr_server.requests[0].headers.get("user-agent", "").startswith("litellm/") is not rust_enabled
