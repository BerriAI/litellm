from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.contracts import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.mark.xfail(
    strict=True,
    reason="public litellm.ocr does not dispatch to the native OCR bridge yet",
)
def test_public_ocr_entrypoint_uses_native_transport_when_enabled(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"
