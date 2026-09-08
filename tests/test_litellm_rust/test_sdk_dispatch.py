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


def test_public_ocr_entrypoint_uses_native_transport_when_enabled(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_public_aocr_entrypoint_uses_native_transport_when_enabled(ocr_server: RecordingServer) -> None:
    response: Final = await litellm.aocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"


def test_public_ocr_falls_back_when_native_transport_declines(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] != "identity"


def test_public_ocr_uses_python_transport_when_disabled(ocr_server: RecordingServer) -> None:
    litellm.rust(False)

    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] != "identity"
