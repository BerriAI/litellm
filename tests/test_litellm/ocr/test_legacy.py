import importlib
from collections.abc import AsyncGenerator
from datetime import datetime
from io import BytesIO
from typing import Final
from unittest.mock import Mock

import httpx
import orjson
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.ocr_lifecycle import NATIVE_OCR_LIFECYCLE


@pytest.fixture
async def provider(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Mock]:
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "0")
    monkeypatch.setattr(bindings, "get_native_bridge", Mock(side_effect=AssertionError("Rust must not load")))
    handler: Final = Mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [{"index": 0, "markdown": "parsed document"}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1},
            },
        )
    )
    transport: Final = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            sync_handler: Final = HTTPHandler(client=sync_client)
            async_handler: Final = AsyncHTTPHandler()
            await async_handler.client.aclose()
            async_handler.client = async_client
            monkeypatch.setattr(llm_http_handler, "_get_httpx_client", lambda: sync_handler)
            monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda llm_provider: async_handler)
            yield handler
    NATIVE_OCR_LIFECYCLE.reset()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "sync_async"])
@pytest.mark.parametrize("dispatch", ["disabled", "declined", "unavailable"])
async def test_python_request_response_and_callbacks(
    provider: Mock, monkeypatch: pytest.MonkeyPatch, mode: str, dispatch: str
) -> None:
    class Declined(Exception):
        pass

    if dispatch != "disabled":
        monkeypatch.setenv("LITELLM_RUST", "1")
        NATIVE_OCR_LIFECYCLE.override(Mock(side_effect=Declined()) if dispatch == "declined" else None)
        main: Final = importlib.import_module("litellm.ocr.main")
        monkeypatch.setattr(main, "native_exception_types", lambda: (Declined, RuntimeError))
    logger: Final = Mock(spec=CustomLogger)
    monkeypatch.setattr(litellm, "input_callback", [logger])
    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "file", "file": BytesIO(b"pdf"), "mime_type": "application/pdf"},
        "api_key": "test-key",
        "api_base": "https://ocr.test/v1",
        "timeout": 7.0,
        "pages": [0, 2],
        "include_image_base64": True,
        "extra_headers": {"x-test-header": "preserved"},
    }

    async def call() -> OCRResponse:
        if mode == "async":
            return await litellm.aocr(**arguments)
        if mode == "sync_async":
            from litellm.litellm_core_utils.litellm_logging import Logging

            logging_obj: Final = Logging(
                model=arguments["model"],
                messages=[],
                stream=False,
                call_type="aocr",
                start_time=datetime.now(),
                litellm_call_id="test-call",
                function_id="test-function",
            )
            return await litellm.ocr(**arguments, aocr=True, litellm_logging_obj=logging_obj)
        return litellm.ocr(**arguments)

    response: Final = await call()
    assert response.pages[0].markdown == "parsed document"
    assert response.usage_info.pages_processed == 1
    assert provider.call_count == 1
    request: Final = provider.call_args.args[0]
    assert str(request.url) == "https://ocr.test/v1/ocr"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-test-header"] == "preserved"
    assert request.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}
    assert orjson.loads(request.content) == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,cGRm"},
        "pages": [0, 2],
        "include_image_base64": True,
    }
    assert logger.log_pre_api_call.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_python_provider_errors_keep_public_exception(provider: Mock, asynchronous: bool) -> None:
    provider.return_value = httpx.Response(429, json={"error": "rate limited"})
    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "https://example.com/file.pdf"},
        "api_key": "test-key",
        "api_base": "https://ocr.test/v1",
        "num_retries": 0,
    }

    async def call() -> object:
        if asynchronous:
            return await litellm.aocr(**arguments)
        return litellm.ocr(**arguments)

    with pytest.raises(litellm.RateLimitError) as error:
        await call()
    assert error.value.status_code == 429
    assert error.value.model == "mistral-ocr-latest"
    assert error.value.llm_provider == "mistral"
    assert provider.call_count == 1
