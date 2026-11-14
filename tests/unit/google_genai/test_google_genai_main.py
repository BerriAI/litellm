#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime
from threading import Event
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
import respx
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.google_genai.main import agenerate_content_stream, generate_content_stream
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.llms.vertex_ai.google_genai.transformation import VertexAIGoogleGenAIConfig
from litellm.types.google_genai.main import GenerateContentResponse
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.router import GenericLiteLLMParams


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


_UPLIFT_PROBE: Final = "gemini-regional-uplift-probe"
_UPLIFT_INPUT_RATE: Final = 1e-06
_UPLIFT_OUTPUT_RATE: Final = 2e-06
_UPLIFT_MULTIPLIER: Final = 1.1


@pytest.fixture
def vertex_regional_uplift_rate_card(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rates: Final = {
        "input_cost_per_token": _UPLIFT_INPUT_RATE,
        "output_cost_per_token": _UPLIFT_OUTPUT_RATE,
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, _UPLIFT_PROBE, {**rates, "litellm_provider": "vertex_ai-language-models"})
    monkeypatch.setattr(litellm, "vertex_language_models", {*litellm.vertex_language_models, _UPLIFT_PROBE})
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{_UPLIFT_PROBE}",
        {
            **rates,
            "litellm_provider": "vertex_ai-language-models",
            "regional_endpoint_uplift_multiplier": _UPLIFT_MULTIPLIER,
        },
    )
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


async def _logged_vertex_generate_content_cost(location: str, *, stream: bool) -> float:
    import asyncio
    import datetime

    from litellm.google_genai.main import GenerateContentHelper
    from litellm.google_genai.streaming_iterator import BaseGoogleGenAIGenerateContentStreamingIterator
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.google_genai.main import GenerateContentResponse

    reply: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8},
        "modelVersion": _UPLIFT_PROBE,
    }
    logging_obj: Final = Logging(
        model=f"vertex_ai/{_UPLIFT_PROBE}",
        messages=[],
        stream=stream,
        call_type="agenerate_content_stream" if stream else "agenerate_content",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="uplift-test",
        function_id="uplift-test",
    )
    setup: Final = GenerateContentHelper.setup_generate_content_call(
        model=f"vertex_ai/{_UPLIFT_PROBE}",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        litellm_logging_obj=logging_obj,
        vertex_location=location,
        stream=stream,
    )
    if stream:
        iterator: Final = BaseGoogleGenAIGenerateContentStreamingIterator(
            litellm_logging_obj=logging_obj,
            request_body={},
            model=setup.model,
            custom_llm_provider=setup.custom_llm_provider,
        )
        iterator.collected_chunks.append(f"data: {json.dumps(reply)}\n\n".encode())
        await iterator._handle_async_streaming_logging()
        await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
    else:
        logging_obj._process_hidden_params_and_response_cost(
            GenerateContentResponse(**reply),
            datetime.datetime(2026, 1, 1),
            datetime.datetime(2026, 1, 1),
            build_logging_payload=False,
        )
    return logging_obj.model_call_details["response_cost"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("location, multiplier", [("global", 1.0), ("us", _UPLIFT_MULTIPLIER)])
async def test_vertex_generate_content_spend_follows_the_deployment_location(
    vertex_regional_uplift_rate_card: None, stream: bool, location: str, multiplier: float
) -> None:
    expected: Final = (7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE) * multiplier

    recorded: Final = await _logged_vertex_generate_content_cost(location, stream=stream)

    assert recorded == pytest.approx(expected), (location, stream)


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
        secret_fields={"raw_headers": {"authorization": "Bearer sk-caller-key"}},
        metadata={"user_api_key_alias": "caller"},
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert result == native_content_handler.response
    received: Final = native_content_handler.requests[0]
    assert received["model"] == "image-model"
    assert received["contents"] is contents
    assert received["config"] == config
    assert received["tools"] is tools
    optional_params: Final = received["optional_params"]
    assert isinstance(optional_params, dict)
    assert optional_params["safetySettings"] == []
    assert optional_params["systemInstruction"] == request["systemInstruction"]
    assert "sk-caller-key" not in str(optional_params)
    assert "metadata" not in optional_params
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
    assert result == native_content_handler.response
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


class FailingCompletionContentLLM(CustomLLM):
    def completion(self, **kwargs: object) -> litellm.ModelResponse:
        raise CustomLLMError(status_code=429, message="adapter quota exhausted")

    async def acompletion(self, **kwargs: object) -> litellm.ModelResponse:
        return self.completion(**kwargs)


class ContentFailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: tuple[str, ...] = ()
        self.asynchronous_calls: tuple[str, ...] = ()
        self.logged: Final = Event()

    def log_failure_event(self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object) -> None:
        self.calls = (*self.calls, str(kwargs["call_type"]))
        self.logged.set()

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.asynchronous_calls = (*self.asynchronous_calls, str(kwargs["call_type"]))
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("use_logging_context", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_delivers_per_request_failure_callbacks_once(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch,
    is_async: bool, use_logging_context: bool,
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": FailingCompletionContentLLM()}]
    )
    global_recorder: Final = ContentFailureRecorder()
    request_recorder: Final = ContentFailureRecorder()
    monkeypatch.setattr(litellm, "callbacks", [global_recorder])
    monkeypatch.setattr(litellm, "failure_callback", [global_recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [global_recorder])
    callback_args: Final = (
        {"litellm_logging_obj": Logging(
            model="native_image_test/chat-model", messages=[], stream=False,
            call_type="agenerate_content" if is_async else "generate_content",
            start_time=datetime(2026, 1, 1), litellm_call_id="failure-test", function_id="failure-test",
            dynamic_failure_callbacks=[request_recorder], dynamic_async_failure_callbacks=[request_recorder],
        )}
        if use_logging_context else {"failure_callback": [request_recorder.log_failure_event]}
    )
    request: Final = dict(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "hello"}]}],
        **callback_args,
    )
    if is_async:
        with pytest.raises(litellm.RateLimitError, match="adapter quota exhausted"):
            await litellm.agenerate_content(**request)
    else:
        with pytest.raises(litellm.RateLimitError, match="adapter quota exhausted"):
            litellm.generate_content(**request)
    assert await asyncio.to_thread(request_recorder.logged.wait, 5)
    assert await asyncio.to_thread(global_recorder.logged.wait, 5)
    expected: Final = ("agenerate_content" if is_async else "generate_content",)
    assert request_recorder.calls == expected
    assert global_recorder.calls == expected
    assert request_recorder.asynchronous_calls == (expected if is_async and use_logging_context else ())
    assert global_recorder.asynchronous_calls == (expected if is_async else ())


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
    assert isinstance(result, dict)
    assert {key: value for key, value in result.items() if key != "_hidden_params"} == native_content_handler.response
    assert "_hidden_params" not in native_content_handler.response
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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_native_config_cannot_overwrite_accounting_context(
    native_content_handler: NativeContentLLM, is_async: bool
) -> None:
    config: Final = {
        "litellm_params": {"metadata": {}},
        "model": "spoofed",
        "custom_llm_provider": "spoofed",
        "call_type": "spoofed",
        "seed": 17,
    }
    metadata: Final = {"user_api_key": "trusted-key-id", "user_api_key_team_id": "trusted-team"}
    request: Final = dict(model="native_image_test/image-model", contents=[], config=config, metadata=metadata)
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    received: Final = native_content_handler.requests[0]
    logged: Final = received["logging_obj"].model_call_details
    assert result == native_content_handler.response
    assert received["config"] == config
    assert logged["litellm_params"]["metadata"]["user_api_key"] == "trusted-key-id"
    assert logged["litellm_params"]["metadata"]["user_api_key_team_id"] == "trusted-team"
    assert logged["model"] == "image-model"
    assert logged["custom_llm_provider"] == "native_image_test"
    assert logged["call_type"] == ("agenerate_content" if is_async else "generate_content")


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_builtin_native_content_keeps_priority_over_colliding_custom_handler(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "gemini", "custom_handler": native_content_handler}])
    litellm.utils.custom_llm_setup()
    request: Final = dict(
        model="gemini/test-model",
        contents=[{"parts": [{"text": "draw"}]}],
        config={"responseModalities": ["IMAGE"]},
        api_key="test-key",
    )
    with respx.mock:
        route: Final = respx.post(
            "https://generativelanguage.googleapis.com/v1alpha/models/test-model:generateContent"
        ).respond(200, json=native_content_handler.response)
        if is_async:
            client: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
            try:
                result = await litellm.agenerate_content(**request, client=client)
            finally:
                await client.close()
        else:
            result = litellm.generate_content(**request)
    assert result.candidates == native_content_handler.response["candidates"]
    assert result.usageMetadata == native_content_handler.response["usageMetadata"]
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)["generationConfig"] == request["config"]
    assert native_content_handler.requests == ()


class ContentSuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.sync_calls: tuple[str, ...] = ()
        self.async_calls: tuple[str, ...] = ()
        self.call_ids: tuple[str, ...] = ()
        self.sync_logged = Event()
        self.async_logged = asyncio.Event()

    def log_success_event(self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object) -> None:
        self.sync_calls = (*self.sync_calls, str(kwargs["call_type"]))
        self.call_ids = (*self.call_ids, str(kwargs["litellm_call_id"]))
        self.sync_logged.set()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.async_calls = (*self.async_calls, str(kwargs["call_type"]))
        self.call_ids = (*self.call_ids, str(kwargs["litellm_call_id"]))
        self.async_logged.set()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_accounts_for_one_success(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": CompletionContentLLM()}]
    )
    recorder: Final = ContentSuccessRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    outer_logging: Final = Logging(
        model="native_image_test/chat-model", messages=[], stream=False,
        call_type="agenerate_content" if is_async else "generate_content",
        start_time=datetime(2026, 1, 1), litellm_call_id="adapter-accounting", function_id="adapter-accounting",
    )
    request: Final = dict(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "hello"}]}],
        litellm_logging_obj=outer_logging,
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    if is_async:
        await asyncio.wait_for(recorder.async_logged.wait(), timeout=5)
        await GLOBAL_LOGGING_WORKER.flush()
        await outer_logging.async_success_handler(result=result)
        assert recorder.async_calls == ("agenerate_content",)
    else:
        assert await asyncio.to_thread(recorder.sync_logged.wait, 5)
        outer_logging.success_handler(result=result)
        assert recorder.sync_calls == ("generate_content",)
    assert result["candidates"][0]["content"]["parts"] == [{"text": "adapter result"}]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_delivers_per_request_success_callbacks_once(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": CompletionContentLLM()}]
    )
    global_recorder: Final = ContentSuccessRecorder()
    request_recorder: Final = ContentSuccessRecorder()
    monkeypatch.setattr(litellm, "callbacks", [global_recorder])
    monkeypatch.setattr(litellm, "success_callback", [global_recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [global_recorder])
    request: Final = dict(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "hello"}]}],
        success_callback=[request_recorder.async_log_success_event if is_async else request_recorder.log_success_event],
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    if is_async:
        await asyncio.wait_for(
            asyncio.gather(request_recorder.async_logged.wait(), global_recorder.async_logged.wait()), timeout=5
        )
        await GLOBAL_LOGGING_WORKER.flush()
        assert request_recorder.async_calls == ("agenerate_content",)
        assert global_recorder.async_calls == ("agenerate_content",)
    else:
        assert await asyncio.to_thread(request_recorder.sync_logged.wait, 5)
        assert await asyncio.to_thread(global_recorder.sync_logged.wait, 5)
        assert request_recorder.sync_calls == ("generate_content",)
        assert global_recorder.sync_calls == ("generate_content",)
    assert result["candidates"][0]["content"]["parts"] == [{"text": "adapter result"}]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_stream_delivers_per_request_success_callbacks_once(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": StreamingContentLLM()}]
    )
    request_recorder: Final = ContentSuccessRecorder()
    callback: Final = request_recorder.async_log_success_event if is_async else request_recorder.log_success_event
    request: Final = dict(model="native_image_test/chat-model", contents=[], success_callback=[callback])
    stream: Final = await agenerate_content_stream(**request) if is_async else generate_content_stream(**request)
    chunks: Final = [chunk async for chunk in stream]
    assert any(b"stream adapter" in chunk for chunk in chunks)
    if is_async:
        await asyncio.wait_for(request_recorder.async_logged.wait(), timeout=5)
        await GLOBAL_LOGGING_WORKER.flush()
        assert request_recorder.async_calls == ("agenerate_content_stream",)
    else:
        assert await asyncio.to_thread(request_recorder.sync_logged.wait, 5)
        assert request_recorder.sync_calls == ("generate_content_stream",)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_logs_under_the_callers_call_id(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool, stream: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": StreamingContentLLM()}]
    )
    recorder: Final = ContentSuccessRecorder()
    callback: Final = recorder.async_log_success_event if is_async else recorder.log_success_event
    request: Final = dict(
        model="native_image_test/chat-model", contents=[], success_callback=[callback], litellm_call_id="client-call"
    )
    if stream:
        stream_response: Final = (
            await agenerate_content_stream(**request) if is_async else generate_content_stream(**request)
        )
        _ = [chunk async for chunk in stream_response]
    else:
        _ = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    if is_async:
        await asyncio.wait_for(recorder.async_logged.wait(), timeout=5)
        await GLOBAL_LOGGING_WORKER.flush()
    else:
        assert await asyncio.to_thread(recorder.sync_logged.wait, 5)
    assert recorder.call_ids == ("client-call",)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_stream_without_stream_hook_fails_before_streaming(
    native_content_handler: NativeContentLLM, is_async: bool
) -> None:
    request: Final = dict(model="native_image_test/image-model", contents=[])
    with pytest.raises(litellm.APIConnectionError, match="astreaming" if is_async else "streaming") as error:
        _ = await agenerate_content_stream(**request) if is_async else generate_content_stream(**request)
    assert error.value.status_code == 500
    assert native_content_handler.requests == ()


@pytest.mark.asyncio
async def test_custom_content_stream_without_stream_hook_keeps_router_legs_serving_chat(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": CompletionContentLLM()}]
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "chat-group",
                "litellm_params": {"model": f"native_image_test/{leg}", "order": order},
                "model_info": {"id": leg},
            }
            for order, leg in enumerate(("leg-a", "leg-b"), start=1)
        ],
        num_retries=0,
        allowed_fails=0,
    )
    for _ in range(2):
        with pytest.raises(litellm.APIConnectionError, match="astreaming"):
            _ = await router.agenerate_content_stream(model="chat-group", contents=[{"parts": [{"text": "hi"}]}])
    response: Final = await router.acompletion(model="chat-group", messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "adapter result"


class CamelCaseContentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[dict[str, object], ...]
    usageMetadata: dict[str, int]
    promptFeedback: dict[str, object] | None = None


_IMAGE_CANDIDATE: Final = {
    "content": {"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]},
    "finishReason": "STOP",
    "index": 0,
}
_IMAGE_USAGE: Final = {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8}
_IMAGE_WIRE_BODY: Final = {"candidates": [_IMAGE_CANDIDATE], "usageMetadata": _IMAGE_USAGE}
_DICT_WITHOUT_USAGE: Final = {"candidates": [_IMAGE_CANDIDATE]}
_NULL_TOOL_ARGS: Final = {"end_date": None, "trip": {"return_date": None, "seats": [None, 2]}}
_HOOK_RESPONSE_SHAPES: Final = (
    pytest.param(
        GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"inline_data": {"mime_type": "image/png", "data": b"image"}}]},
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            usage_metadata={"prompt_token_count": 3, "candidates_token_count": 5, "total_token_count": 8},
        ),
        (3, 5),
        _IMAGE_WIRE_BODY,
        id="snake-case-sdk-object-with-bytes",
    ),
    pytest.param(
        CamelCaseContentResponse(candidates=(_IMAGE_CANDIDATE,), usageMetadata=_IMAGE_USAGE),
        (3, 5),
        _IMAGE_WIRE_BODY,
        id="camel-case-model-with-unset-fields",
    ),
    pytest.param(
        GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"function_call": {"name": "book", "args": _NULL_TOOL_ARGS}}]},
                    "index": 0,
                }
            ],
            usage_metadata={"prompt_token_count": 3, "candidates_token_count": 5, "total_token_count": 8},
        ),
        (3, 5),
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"functionCall": {"name": "book", "args": _NULL_TOOL_ARGS}}]},
                    "index": 0,
                }
            ],
            "usageMetadata": _IMAGE_USAGE,
        },
        id="sdk-object-with-null-tool-args",
    ),
    pytest.param(
        GenerateContentResponse(
            candidates=[_IMAGE_CANDIDATE],
            usage_metadata={"prompt_token_count": 3, "candidates_token_count": 5, "total_token_count": 8},
            sdk_http_response={"headers": {"x-upstream-trace": "abc"}},
            automatic_function_calling_history=[{"role": "user", "parts": [{"text": "draw"}]}],
            parsed={"answer": "draw"},
        ),
        (3, 5),
        _IMAGE_WIRE_BODY,
        id="sdk-object-with-client-side-fields",
    ),
    pytest.param(
        GenerateContentResponse(
            candidates=[_IMAGE_CANDIDATE],
            usage_metadata={"prompt_token_count": 3, "candidates_token_count": 5, "total_token_count": 8},
            sdk_http_response={"headers": {"x-upstream-trace": "abc"}},
            automatic_function_calling_history=[{"role": "user", "parts": [{"text": "draw"}]}],
            parsed={"answer": "draw"},
        ).model_dump(exclude_none=True),
        (3, 5),
        _IMAGE_WIRE_BODY,
        id="dumped-sdk-dict-with-client-side-fields",
    ),
    pytest.param(_DICT_WITHOUT_USAGE, (0, 0), _DICT_WITHOUT_USAGE, id="dict-without-usage"),
)


class ContentUsageRecorder:
    def __init__(self) -> None:
        self.usage: tuple[tuple[object, object], ...] = ()
        self.costs: tuple[object, ...] = ()
        self.logged: Final = Event()

    def log(self, kwargs: dict[str, object], *_: object) -> None:
        standard_logging_object: Final = kwargs["standard_logging_object"]
        assert isinstance(standard_logging_object, dict)
        self.usage = (
            *self.usage,
            (standard_logging_object["prompt_tokens"], standard_logging_object["completion_tokens"]),
        )
        self.costs = (*self.costs, standard_logging_object["response_cost"])
        self.logged.set()

    async def alog(self, kwargs: dict[str, object], *rest: object) -> None:
        self.log(kwargs, *rest)


@pytest.mark.parametrize("hook_response, expected_usage, expected_result", _HOOK_RESPONSE_SHAPES)
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_native_content_logs_each_hook_response_shape_once(
    native_content_handler: NativeContentLLM,
    is_async: bool,
    hook_response: object,
    expected_usage: tuple[int, int],
    expected_result: dict[str, object],
) -> None:
    native_content_handler.response = hook_response
    recorder: Final = ContentUsageRecorder()
    request: Final = dict(
        model="native_image_test/image-model", contents=[{"parts": [{"text": "draw"}]}],
        success_callback=[recorder.alog if is_async else recorder.log],
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert await asyncio.to_thread(recorder.logged.wait, 5)
    if is_async:
        await GLOBAL_LOGGING_WORKER.flush()
    assert result == expected_result
    assert recorder.usage == (expected_usage,)


@pytest.mark.asyncio
async def test_router_tags_custom_native_content_objects_with_their_deployment(
    native_content_handler: NativeContentLLM,
) -> None:
    native_content_handler.response = CamelCaseContentResponse(candidates=(_IMAGE_CANDIDATE,), usageMetadata=_IMAGE_USAGE)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "image-group",
                "litellm_params": {"model": "native_image_test/image-model"},
                "model_info": {"id": "native-leg"},
            }
        ]
    )
    result: Final = await router.agenerate_content(model="image-group", contents=[], config={})
    assert isinstance(result, dict)
    hidden_params: Final = result.pop("_hidden_params")
    assert result == _IMAGE_WIRE_BODY
    assert hidden_params["model_id"] == "native-leg"
    assert hidden_params["additional_headers"]["x-litellm-model-group"] == "image-group"


class AdapterOnlyContentLLM(CompletionContentLLM):
    streaming = StreamingContentLLM.streaming
    astreaming = StreamingContentLLM.astreaming


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_rejects_generation_config_it_cannot_send(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch,
    is_async: bool, stream: bool, drop_params: bool,
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": AdapterOnlyContentLLM()}]
    )
    request: Final = dict(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "draw"}]}],
        config={"temperature": 0.2, "imageConfig": {"aspectRatio": "16:9"}}, drop_params=drop_params,
    )

    async def call() -> object:
        if stream:
            response: Final = (
                await agenerate_content_stream(**request) if is_async else generate_content_stream(**request)
            )
            return [chunk async for chunk in response]
        return await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)

    if not drop_params:
        with pytest.raises(litellm.BadRequestError, match="imageConfig"):
            _ = await call()
        return
    result: Final = await call()
    if stream:
        assert b"stream adapter" in b"".join(result)
    else:
        assert result["candidates"][0]["content"]["parts"] == [{"text": "adapter result"}]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_accepts_generation_config_it_already_satisfies(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": AdapterOnlyContentLLM()}]
    )
    request: Final = dict(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "hi"}]}],
        config={"responseModalities": ["TEXT"], "candidateCount": 1},
    )
    result: Final = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert result["candidates"][0]["content"]["parts"] == [{"text": "adapter result"}]


@pytest.mark.parametrize(
    "config",
    [
        {"responseModalities": ["TEXT", "IMAGE"]},
        {"candidateCount": 2},
        {"responseMimeType": "application/json"},
        {"responseMimeType": "application/json", "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}}},
        {"responseMimeType": "text/x.enum", "responseSchema": {"type": "STRING", "enum": ["a", "b"]}},
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_rejects_generation_config_values_it_cannot_satisfy(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch,
    is_async: bool, config: dict[str, object],
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": AdapterOnlyContentLLM()}]
    )
    request: Final = dict(model="native_image_test/chat-model", contents=[{"parts": [{"text": "hi"}]}], config=config)
    with pytest.raises(litellm.BadRequestError, match=next(iter(config))):
        _ = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)


class ResponseFormatRecorderLLM(CompletionContentLLM):
    def __init__(self) -> None:
        super().__init__()
        self.response_formats: tuple[object, ...] = ()

    def completion(self, **kwargs: object) -> litellm.ModelResponse:
        optional_params: Final = kwargs["optional_params"]
        assert isinstance(optional_params, dict)
        self.response_formats = (*self.response_formats, optional_params.get("response_format"))
        return super().completion(**kwargs)


@pytest.mark.parametrize(
    ("config", "response_format"),
    [
        ({"responseMimeType": "text/plain"}, None),
        (
            {
                "responseMimeType": "application/json",
                "responseSchema": {"type": "OBJECT", "properties": {"color": {"type": "STRING"}}},
            },
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "properties": {"color": {"type": "string"}}},
                },
            },
        ),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_content_adapter_sends_the_response_format_it_accepts(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch,
    is_async: bool, config: dict[str, object], response_format: object,
) -> None:
    handler: Final = ResponseFormatRecorderLLM()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": handler}])
    request: Final = dict(model="native_image_test/chat-model", contents=[{"parts": [{"text": "hi"}]}], config=config)
    _ = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert handler.response_formats == (response_format,)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_custom_native_content_bills_per_request_prices(
    native_content_handler: NativeContentLLM, is_async: bool
) -> None:
    recorder: Final = ContentUsageRecorder()
    request: Final = dict(
        model="native_image_test/image-model", contents=[{"parts": [{"text": "draw"}]}],
        input_cost_per_token=0.5, output_cost_per_token=2.0,
        success_callback=[recorder.alog if is_async else recorder.log],
    )
    _ = await litellm.agenerate_content(**request) if is_async else litellm.generate_content(**request)
    assert await asyncio.to_thread(recorder.logged.wait, 5)
    if is_async:
        await GLOBAL_LOGGING_WORKER.flush()
    assert recorder.usage == ((3, 5),)
    assert recorder.costs == (pytest.approx(3 * 0.5 + 5 * 2.0),)


@pytest.mark.asyncio
async def test_custom_content_adapter_holds_success_logs_while_the_caller_defers_them(
    native_content_handler: NativeContentLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "native_image_test", "custom_handler": CompletionContentLLM()}]
    )
    recorder: Final = ContentSuccessRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    outer_logging: Final = Logging(
        model="native_image_test/chat-model", messages=[], stream=False, call_type="agenerate_content",
        start_time=datetime(2026, 1, 1), litellm_call_id="deferred-adapter", function_id="deferred-adapter",
    )
    outer_logging.defer_async_logging = True
    _ = await litellm.agenerate_content(
        model="native_image_test/chat-model", contents=[{"parts": [{"text": "hello"}]}],
        litellm_logging_obj=outer_logging,
    )
    await GLOBAL_LOGGING_WORKER.flush()
    assert (recorder.sync_calls, recorder.async_calls) == ((), ())
    enqueue_deferred_logging: Final = outer_logging.enqueue_deferred_logging
    assert enqueue_deferred_logging is not None
    enqueue_deferred_logging()
    await asyncio.wait_for(recorder.async_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()
    assert recorder.async_calls == ("agenerate_content",)
    assert set(recorder.call_ids) == {"deferred-adapter"}


_LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
_IDENTITY: Final = MappingProxyType({"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"})
_PROXY_METADATA: Final = MappingProxyType(
    {**_IDENTITY, "spend_logs_metadata": {"cost-center": "cc-1"}, "requester_metadata": {"team": "search"}}
)
_FORGED: Final = MappingProxyType({"user_api_key_alias": "forged", "app": "bot"})
_HANDLER_CALLS: Final = pytest.mark.parametrize(
    ("is_async", "stream"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["sync", "async", "sync-stream", "async-stream"],
)
_GEMINI_REPLY: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}
_SENT_BODY: Final = TypeAdapter(Mapping[str, object])
_SENT_LABELS: Final = TypeAdapter(Mapping[str, str])


class _SignedInVertexGoogleGenAIConfig(VertexAIGoogleGenAIConfig):
    def _ensure_access_token(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> tuple[str, str]:
        return "token", "proj"

    async def _ensure_access_token_async(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> tuple[str, str]:
        return "token", "proj"


async def _generate_content_body(
    config: GoogleGenAIConfig,
    litellm_params: GenericLiteLLMParams,
    extra_body: Mapping[str, object],
    *,
    is_async: bool,
    stream: bool,
) -> Mapping[str, object]:
    import datetime
    from unittest.mock import MagicMock

    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    reply: Final = MagicMock(side_effect=lambda _request: httpx.Response(200, json=_GEMINI_REPLY))
    transport: Final = httpx.MockTransport(reply)
    client: Final = (
        AsyncHTTPHandler(transport=transport) if is_async else HTTPHandler(client=httpx.Client(transport=transport))
    )
    call: Final = BaseLLMHTTPHandler().generate_content_handler(
        model="gemini-3.8-flash",
        contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
        generate_content_provider_config=config,
        generate_content_config_dict={},
        tools=None,
        custom_llm_provider=config.custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=Logging(
            model="gemini-3.8-flash",
            messages=[],
            stream=stream,
            call_type="agenerate_content" if is_async else "generate_content",
            start_time=datetime.datetime(2026, 1, 1),
            litellm_call_id="labels-test",
            function_id="labels-test",
        ),
        extra_body=dict(extra_body),
        _is_async=is_async,
        client=client,
        stream=stream,
    )
    if is_async:
        await call

    sent: Final[httpx.Request] = reply.call_args.args[0]
    return _SENT_BODY.validate_json(sent.content)


async def _vertex_generate_content_body(
    extra_body: Mapping[str, object], *, is_async: bool, stream: bool
) -> Mapping[str, object]:
    return await _generate_content_body(
        _SignedInVertexGoogleGenAIConfig(),
        GenericLiteLLMParams(metadata=dict(_PROXY_METADATA), vertex_location="us-central1"),
        extra_body,
        is_async=is_async,
        stream=stream,
    )


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_vertex_generate_content_sends_identity_labels_a_caller_cannot_forge(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _vertex_generate_content_body({"labels": dict(_FORGED)}, is_async=is_async, stream=stream)

    labels: Final = _SENT_LABELS.validate_python(body["labels"])
    assert list(labels.items()) == [*_IDENTITY.items(), ("app", "bot"), ("cost-center", "cc-1")]


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_vertex_generate_content_without_caller_labels_still_carries_identity(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _vertex_generate_content_body({}, is_async=is_async, stream=stream)

    labels: Final = _SENT_LABELS.validate_python(body["labels"])
    assert list(labels.items()) == [*_IDENTITY.items(), ("team", "search"), ("cost-center", "cc-1")]


@pytest.mark.asyncio
@_HANDLER_CALLS
@pytest.mark.parametrize(
    "extra_body, expected_labels",
    [({}, "absent"), ({"labels": dict(_FORGED)}, _FORGED), ({"labels": "not-a-mapping"}, "not-a-mapping")],
    ids=["no-labels", "labels", "non-mapping-labels"],
)
async def test_vertex_generate_content_with_the_setting_off_sends_only_the_callers_labels(
    is_async: bool,
    stream: bool,
    extra_body: Mapping[str, object],
    expected_labels: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)

    body: Final = await _vertex_generate_content_body(extra_body, is_async=is_async, stream=stream)

    assert body.get("labels", "absent") == expected_labels


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_ai_studio_generate_content_keeps_the_callers_labels_with_the_setting_on(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _generate_content_body(
        GoogleGenAIConfig(),
        GenericLiteLLMParams(metadata=dict(_PROXY_METADATA), api_key="test-key"),
        {"labels": dict(_FORGED)},
        is_async=is_async,
        stream=stream,
    )

    assert body["labels"] == _FORGED
