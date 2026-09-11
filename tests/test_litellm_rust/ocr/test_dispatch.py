from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import main as ocr_main
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
    monkeypatch: pytest.MonkeyPatch,
    rust_enabled: bool,
) -> None:
    rust_call: Final = Mock(wraps=ocr_main.rust_ocr_bridge.ocr)
    python_call: Final = Mock(wraps=ocr_main.base_llm_http_handler.ocr)
    monkeypatch.setattr(ocr_main.rust_ocr_bridge, "ocr", rust_call)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", python_call)
    litellm.rust(rust_enabled)

    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "native OCR response"
    assert rust_call.call_count == int(rust_enabled)
    assert python_call.call_count == int(not rust_enabled)
    assert len(ocr_server.requests) == 1
