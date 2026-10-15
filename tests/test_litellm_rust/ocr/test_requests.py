from io import BytesIO
from typing import Final

import pytest

from tests.test_litellm_rust.ocr.data import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_aocr,
    call_ocr,
)
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("model", ["mistral/mistral-ocr-latest", "azure_ai/mistral-ocr-latest", "reducto/parse-v3"])
async def test_ocr_contract_native_format_supported(
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    model: str,
) -> None:
    server.expected_requests = None
    payload: Final = (
        {"result": {"chunks": [{"content": "native OCR response"}]}, "usage": {"num_pages": 1}}
        if model.startswith("reducto/")
        else OCR_RESPONSE
    )
    server.default_response = ResponseSpec(body=payload)
    arguments: Final = {
        "model": model,
        "req_format": "native",
        "num_retries": 0,
        "document": {"type": "document_url", "document_url": "reducto://ready.pdf"}
        if model.startswith("reducto/")
        else OCR_DOCUMENT,
    }
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert response.get_provider_native_response() == payload
    assert len(server.requests) == 1
    if ocr_backend:
        assert_served_natively(server, response)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ocr_contract_unknown_reducto_model_reaches_provider(
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    server.default_response = ResponseSpec(body={"result": {"chunks": [{"content": "future model response"}]}})
    arguments: Final = {
        "model": "reducto/future-parse-model",
        "document": {"type": "document_url", "document_url": "reducto://ready.pdf"},
        "num_retries": 0,
    }
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)
    assert response.model == "future-parse-model"
    assert response.pages[0].markdown == "future model response"
    assert len(server.requests) == 1
    assert server.requests[0].path == "/parse"
    assert server.requests[0].body == {"input": "reducto://ready.pdf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_uses_token_provider_result_as_bearer_token(
    server: RecordingServer, isolated_azure_auth: None, asynchronous: bool
) -> None:
    calls: Final = []

    def token_provider() -> str:
        calls.append("token")
        return "callback-token"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": token_provider,
    }
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)

    assert calls == ["token"]
    assert response.pages[0].markdown == "native OCR response"
    assert_served_natively(server, response)
    assert server.requests[0].headers["authorization"] == "Bearer callback-token"


def test_native_ocr_encodes_python_file_input_and_drops_unknown_arguments(server: RecordingServer) -> None:
    response: Final = call_ocr(
        server,
        document={"type": "file", "file": BytesIO(b"abc"), "mime_type": "image/png"},
        opaque_extension=object(),
    )

    assert response.pages[0].markdown == "native OCR response"
    assert_served_natively(server, response)
    assert server.requests[0].body == {
        "model": "mistral-ocr-latest",
        "document": {"type": "image_url", "image_url": "data:image/png;base64,YWJj"},
    }
