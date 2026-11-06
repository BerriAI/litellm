#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest




import litellm
from litellm.google_genai.main import agenerate_content_stream, generate_content_stream
from litellm.llms.custom_llm import CustomLLM, CustomLLMError


@pytest.mark.asyncio
async def test_agenerate_content_stream():
    """
    Test that the agenerate_content_stream function works
    """
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import (
        agenerate_content_stream,
        base_llm_http_handler,
    )

    with patch.object(
        base_llm_http_handler, "generate_content_handler", new=AsyncMock()
    ) as mock_post:
        result = await agenerate_content_stream(
            model="gemini/gemini-2.0-flash-001",
            contents="Hello, world!",
            stream=True,
        )
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["stream"] is True


def _mock_gemini_post_response():
    """A minimal stand-in for a successful Gemini generateContent HTTP response."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hi"}], "role": "model"},
                "finishReason": "STOP",
            }
        ]
    }
    return resp


NATIVE_TOP_LEVEL_FIELD_CASES = [
    (
        "safetySettings",
        [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
    ),
    ("toolConfig", {"functionCallingConfig": {"mode": "AUTO"}}),
    ("cachedContent", "cachedContents/abc123"),
    ("labels", {"team": "search"}),
    ("continuationToken", "dG9rLTE="),
]


@pytest.mark.parametrize("field_name, field_value", NATIVE_TOP_LEVEL_FIELD_CASES)
def test_native_top_level_field_forwarded_to_request_body(field_name, field_value):
    """
    Regression for https://github.com/BerriAI/litellm/issues/12671

    Google's native generateContent body carries fields like safetySettings at the
    top level (siblings of generationConfig). The proxy spreads them as loose kwargs
    into generate_content. They must reach Google's request body at the top level and
    must NOT be silently dropped nor nested under generationConfig.
    """
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            **{field_name: field_value},
        )

    assert mock_post.called, "expected the request to reach the HTTP client"
    body = mock_post.call_args.kwargs["json"]
    assert (
        body[field_name] == field_value
    ), f"{field_name} should be forwarded to Google at the top level"
    assert field_name not in body.get("generationConfig", {}), (
        f"{field_name} must be a top-level sibling of generationConfig, "
        "not nested inside it"
    )


@pytest.mark.asyncio
async def test_native_safety_settings_forwarded_async():
    """The async path (used by the proxy's :generateContent route) must also forward
    native top-level fields."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})


def test_native_fields_coexist_with_generation_config():
    """Forwarding native top-level fields must not regress the already-working
    generationConfig path; both must land in their correct positions."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            generationConfig={"temperature": 0, "responseMimeType": "application/json"},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    generation_config = body["generationConfig"]
    assert generation_config["temperature"] == 0
    assert generation_config["responseMimeType"] == "application/json"


def test_explicit_extra_body_overrides_native_top_level_field():
    """An explicit extra_body value takes precedence over the same top-level field."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    native = [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
    override = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=native,
            extra_body={"safetySettings": override},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == override


def test_native_fields_and_system_instruction_forwarded_on_sync_stream():
    """The sync streaming path (generate_content_stream) must forward native top-level
    fields AND systemInstruction. The PR changed the merge here and newly added the
    systemInstruction kwarg; without coverage a regression on either ships green."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content_stream
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]
    system_instruction = {"parts": [{"text": "Be terse"}]}

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            systemInstruction=system_instruction,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert body["systemInstruction"] == system_instruction
    assert "safetySettings" not in body.get("generationConfig", {})


@pytest.mark.asyncio
async def test_native_fields_forwarded_on_async_stream():
    """The async streaming path (agenerate_content_stream) backs the proxy's
    :streamGenerateContent route and must forward native top-level fields too."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content_stream
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})


@pytest.mark.parametrize("base_model, uses_native_api", [("gemini-2.5-flash", True), (None, False)])
def test_vertex_generate_content_routes_by_deployment_base_model(base_model, uses_native_api):
    import datetime

    from litellm.google_genai.main import GenerateContentHelper
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="vertex_ai/openai/tuned-endpoint",
        messages=[],
        stream=False,
        call_type="agenerate_content",
        start_time=datetime.datetime.now(),
        litellm_call_id="route-test",
        function_id="route-test",
    )
    setup = GenerateContentHelper.setup_generate_content_call(
        model="vertex_ai/openai/tuned-endpoint",
        contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
        litellm_logging_obj=logging_obj,
        **({} if base_model is None else {"base_model": base_model}),
    )

    assert (setup.generate_content_provider_config is not None) is uses_native_api


class NativeContentLLM(CustomLLM):
    def __init__(self) -> None:
        super().__init__()
        self.requests: tuple[dict[str, object], ...] = ()
        self.response: dict[str, object] = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}],
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
        }

    def generate_content(self, **kwargs: object) -> dict[str, object]:
        self.requests = (*self.requests, kwargs)
        return self.response

    async def agenerate_content(self, **kwargs: object) -> dict[str, object]:
        self.requests = (*self.requests, kwargs)
        return self.response


class CompletionContentLLM(CustomLLM):
    def completion(self, **kwargs: object) -> litellm.ModelResponse:
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "adapter result"}}])

    async def acompletion(self, **kwargs: object) -> litellm.ModelResponse:
        return self.completion(**kwargs)


class StreamingContentLLM(NativeContentLLM):
    def streaming(self, **kwargs: object) -> Iterator[litellm.GenericStreamingChunk]:
        yield litellm.GenericStreamingChunk(
            text="stream adapter",
            tool_use=None,
            is_finished=True,
            finish_reason="stop",
            index=0,
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

    async def astreaming(self, **kwargs: object) -> AsyncIterator[litellm.GenericStreamingChunk]:
        for chunk in self.streaming(**kwargs):
            yield chunk


@pytest.fixture
def native_content_handler(monkeypatch: pytest.MonkeyPatch) -> NativeContentLLM:
    handler: Final = NativeContentLLM()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": handler}])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list])
    litellm.utils.custom_llm_setup()
    return handler


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_native_content_preserves_images_and_config(
    native_content_handler: NativeContentLLM, is_async: bool
) -> None:
    contents: Final = [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW5wdXQ="}}]}]
    config: Final = {"imageConfig": {"aspectRatio": "1:1"}, "responseModalities": ["IMAGE"], "seed": 17}
    tools: Final = [{"googleSearch": {}}]
    request: Final = dict(
        model="native_image_test/image-model",
        contents=contents,
        config=config,
        tools=tools,
        safetySettings=[],
        systemInstruction={"parts": [{"text": "draw"}]},
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert result is native_content_handler.response
    received: Final = native_content_handler.requests[0]
    assert received["model"] == "image-model"
    assert received["contents"] is contents
    assert received["config"] == config
    assert received["tools"] is tools
    optional_params: Final = received["optional_params"]
    assert isinstance(optional_params, dict)
    assert optional_params["safetySettings"] == []
    assert optional_params["systemInstruction"] == request["systemInstruction"]
    params: Final = received["litellm_params"]
    assert isinstance(params, litellm.types.router.GenericLiteLLMParams)
    assert params.custom_llm_provider == "native_image_test"
    assert received["logging_obj"] is not None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("config", [None, {}])
@pytest.mark.asyncio
async def test_custom_native_content_preserves_empty_request(
    native_content_handler: NativeContentLLM, is_async: bool, config: object
) -> None:
    request: Final = dict(model="native_image_test/image-model", contents=[], config=config)
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert result is native_content_handler.response
    received: Final = native_content_handler.requests[0]
    assert received["contents"] == []
    assert received["config"] == config
    assert received["tools"] is None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_without_native_hook_uses_completion(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": CompletionContentLLM()}]
    )
    request: Final = dict(model="native_image_test/chat-model", contents=[{"parts": [{"text": "hello"}]}])
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert result["candidates"][0]["content"]["parts"] == [{"text": "adapter result"}]


@pytest.mark.asyncio
async def test_router_custom_native_content_leg(native_content_handler: NativeContentLLM) -> None:
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "image-group", "litellm_params": {"model": "native_image_test/image-model", "order": 0}},
            {
                "model_name": "image-group",
                "litellm_params": {"model": "gemini/test-model", "api_key": "test-key", "order": 1},
            },
        ]
    )
    result: Final = await router.agenerate_content(model="image-group", contents=[], config={})
    assert result == native_content_handler.response
    assert native_content_handler.requests[0]["contents"] == []
    assert native_content_handler.requests[0]["config"] == {}


class FailingContentLLM(CustomLLM):
    def generate_content(self, **kwargs: object) -> dict[str, object]:
        raise CustomLLMError(status_code=429, message="native quota exhausted")

    async def agenerate_content(self, **kwargs: object) -> dict[str, object]:
        raise CustomLLMError(status_code=429, message="native quota exhausted")


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_native_content_maps_errors(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": FailingContentLLM()}]
    )
    if is_async:
        with pytest.raises(litellm.RateLimitError, match="native quota exhausted"):
            await litellm.agenerate_content(model="native_image_test/image-model", contents=[])
    else:
        with pytest.raises(litellm.RateLimitError, match="native quota exhausted"):
            litellm.generate_content(model="native_image_test/image-model", contents=[])


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_stream_uses_completion_adapter(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    handler: Final = StreamingContentLLM()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": handler}])
    stream: Final = (
        await agenerate_content_stream(model="native_image_test/chat-model", contents=[])
        if is_async
        else generate_content_stream(model="native_image_test/chat-model", contents=[])
    )
    chunks: Final = [chunk async for chunk in stream]
    assert any(b"stream adapter" in chunk for chunk in chunks)
    assert handler.requests == ()
