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


@pytest.mark.parametrize("enabled", [False, True, None])
def test_public_ocr_uses_native_route_independently_of_flag(ocr_server: RecordingServer, enabled: bool | None) -> None:
    litellm.rust(enabled)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("caching", [None, False, True])
async def test_ocr_does_not_depend_on_chat_cache(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, caching: bool | None
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local", supported_call_types=["completion", "acompletion"]))
    arguments: Final = {
        "model": OCR_MODEL,
        "document": OCR_DOCUMENT,
        "api_key": "test-key",
        "api_base": ocr_server.base_url,
        "caching": caching,
    }
    response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert len(ocr_server.requests) == 1
