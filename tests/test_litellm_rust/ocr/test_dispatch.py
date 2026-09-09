from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.response_marker import has_rust_response_marker
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def test_sync_ocr_response_is_marked_as_rust_when_native_dispatch_is_enabled(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert has_rust_response_marker(response)


@pytest.mark.asyncio
async def test_async_ocr_response_is_marked_as_rust_when_native_dispatch_is_enabled(
    ocr_server: RecordingServer,
) -> None:
    response: Final = await litellm.aocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert has_rust_response_marker(response)


def test_public_ocr_falls_back_to_python_for_unsupported_file_document(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert not has_rust_response_marker(response)


def test_public_ocr_response_has_no_rust_marker_when_native_dispatch_is_disabled(
    ocr_server: RecordingServer,
) -> None:
    litellm.rust(False)

    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert not has_rust_response_marker(response)


def test_native_ocr_sends_identity_accept_encoding_header(ocr_server: RecordingServer) -> None:
    litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"
