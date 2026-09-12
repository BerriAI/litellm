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


def test_public_ocr_uses_native_route_when_enabled(ocr_server: RecordingServer) -> None:
    litellm.rust(True)
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "native OCR response"
    assert len(ocr_server.requests) == 1
    assert not ocr_server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


def test_public_ocr_fails_before_network_when_native_is_disabled(ocr_server: RecordingServer) -> None:
    litellm.rust(False)
    ocr_server.expected_requests = 0

    with pytest.raises(RuntimeError, match="Rust OCR is unavailable"):
        litellm.ocr(
            model=OCR_MODEL,
            document=OCR_DOCUMENT,
            api_key="test-key",
            api_base=ocr_server.base_url,
        )

    assert ocr_server.requests == []
