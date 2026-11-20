import base64
import importlib
from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.ollama.decisions.transformation import OllamaDecisionsConfig
from litellm.llms.openai.decisions.transformation import openai_request_to_ir
from litellm.types.decisions import DecisionsRequestBody, OpenAIDecisionRequestBody

_IMAGE: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1XkAAAAASUVORK5CYII="
_DATA_URL: Final = f"data:image/png;base64,{_IMAGE}"
_SYSTEMONE: Final = TypeAdapter(DecisionsRequestBody)
_OPENAI: Final = TypeAdapter(OpenAIDecisionRequestBody)
_IMAGES: Final = TypeAdapter(list[str])


@pytest.mark.parametrize("image", (_DATA_URL, {"content_type": "image/png", "base64": _IMAGE}))
def test_systemone_images_decode_as_native_ollama_image_bytes(image: str | dict[str, str]) -> None:
    request: Final = systemone_request_to_ir(
        _SYSTEMONE.validate_python(
            {
                "state": "Read the label.",
                "images": [image],
                "questions": {"label": {"type": "choice", "criteria": {"17": None, "62": None}}},
            }
        )
    )
    body: Final = OllamaDecisionsConfig().transform_decisions_request("clef", request, "ollama")
    images: Final = _IMAGES.validate_python(body["images"])

    assert [base64.b64decode(value, validate=True) for value in images] == [base64.b64decode(_IMAGE)]


def test_openai_image_input_decodes_as_native_ollama_image_bytes() -> None:
    request: Final = openai_request_to_ir(
        _OPENAI.validate_python(
            {
                "input": [{"role": "user", "content": [{"type": "input_image", "image_url": _DATA_URL}]}],
                "questions": [
                    {"type": "choice", "instructions": "Read the label.", "choices": [{"value": "17"}, {"value": "62"}]}
                ],
            }
        )
    )
    body: Final = OllamaDecisionsConfig().transform_decisions_request("clef", request, "ollama")
    images: Final = _IMAGES.validate_python(body["images"])

    assert [base64.b64decode(value, validate=True) for value in images] == [base64.b64decode(_IMAGE)]


class _CountedImageStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.bytes_read = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for offset in range(0, len(self.body), 32):
            chunk = self.body[offset : offset + 32]
            self.bytes_read += len(chunk)
            yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("oversized", (False, True))
async def test_remote_images_enforce_download_limit_before_buffering(monkeypatch, mode: str, oversized: bool) -> None:
    call_module = importlib.import_module("litellm.decisions.call")
    image: Final = base64.b64decode(_IMAGE)
    stream: Final = _CountedImageStream(image + (bytes(16384) if oversized else b""))
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Type": "image/png"}, stream=stream)
    )
    client: Final = httpx.Client(transport=transport)
    aclient: Final = AsyncHTTPHandler(transport=transport)
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=client))
    monkeypatch.setattr(litellm, "module_level_aclient", aclient)
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(image_handling, "in_memory_cache", InMemoryCache(max_size_in_memory=10))
    limit_mb: Final = len(image) / (1024 * 1024)
    monkeypatch.setattr(image_handling, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", limit_mb)
    monkeypatch.setattr(call_module, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", limit_mb, raising=False)
    request: Final = _SYSTEMONE.validate_python(
        {
            "state": "Read the label.",
            "questions": {"label": {"type": "choice", "criteria": {"17": None, "62": None}}},
            "images": ["https://images.example.test/label.png"],
        }
    )
    ir_request: Final = systemone_request_to_ir(request)
    call: Final = call_module.DecisionsCall(
        model="clef",
        requested_model="ollama/clef",
        custom_llm_provider="ollama",
        provider_config=OllamaDecisionsConfig(),
        request=request,
        ir_request=ir_request,
        api_base="https://ollama.example.test",
        api_key=None,
        logging_obj=None,
        headers={},
        timeout=1,
        litellm_params={},
        caller_questions=ir_request.questions,
    )

    async def inline():
        if mode == "sync":
            return call_module._with_remote_images_inlined(call)
        return await call_module._async_with_remote_images_inlined(call)

    try:
        if oversized:
            with pytest.raises(litellm.ImageFetchError):
                await inline()
            assert stream.bytes_read <= len(image) + 32
        else:
            inlined = await inline()
            body = inlined.provider_config.transform_decisions_request(
                inlined.model, inlined.ir_request, inlined.custom_llm_provider
            )
            assert [base64.b64decode(value, validate=True) for value in _IMAGES.validate_python(body["images"])] == [
                image
            ]
        assert stream.closed
    finally:
        client.close()
        await aclient.client.aclose()
