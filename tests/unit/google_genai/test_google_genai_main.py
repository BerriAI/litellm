#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx




import litellm


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


_OUTPUT_CAP_SPELLINGS: Final = (
    {"generationConfig": {"maxOutputTokens": 16}},
    {"generation_config": {"max_output_tokens": 16}},
    {"generationConfig": {"max_output_tokens": 16}},
    {"generation_config": {"maxOutputTokens": 16}},
)
_CHAT_UPSTREAM: Final = "http://upstream.test/v1"
_CHAT_STREAM_BODY: Final = (
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"grok",'
    '"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
)
_CHAT_BODY: Final = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "grok",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _chat_reply(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return httpx.Response(200, text=_CHAT_STREAM_BODY, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_CHAT_BODY)


@pytest.fixture
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


async def _call_entrypoint(entrypoint: str, output_cap: dict[str, dict[str, int]]) -> None:
    from litellm.google_genai.main import (
        agenerate_content,
        agenerate_content_stream,
        generate_content,
        generate_content_stream,
    )

    request: Final = {
        "model": "hosted_vllm/grok",
        "contents": [{"role": "user", "parts": [{"text": "Write a story"}]}],
        "api_base": _CHAT_UPSTREAM,
        "api_key": "k",
        **output_cap,
    }
    match entrypoint:
        case "generate_content":
            generate_content(**request)
        case "agenerate_content":
            await agenerate_content(**request)
        case "generate_content_stream":
            generate_content_stream(**request)
        case "agenerate_content_stream":
            async for _ in await agenerate_content_stream(**request):
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("output_cap", _OUTPUT_CAP_SPELLINGS)
@pytest.mark.parametrize(
    "entrypoint", ["generate_content", "agenerate_content", "generate_content_stream", "agenerate_content_stream"]
)
async def test_adapter_path_sends_the_output_cap_in_every_spelling(
    _httpx_transport: None, entrypoint: str, output_cap: dict[str, dict[str, int]]
) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_CHAT_UPSTREAM}/chat/completions").mock(side_effect=_chat_reply)
        await _call_entrypoint(entrypoint, output_cap)

    assert json.loads(route.calls.last.request.content)["max_tokens"] == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("output_cap", _OUTPUT_CAP_SPELLINGS)
async def test_native_gemini_request_carries_the_output_cap_in_every_spelling(
    _httpx_transport: None, output_cap: dict[str, dict[str, int]]
) -> None:
    from litellm.google_genai.main import agenerate_content

    with respx.mock() as upstream:
        route: Final = upstream.post(url__startswith="https://generativelanguage.googleapis.com/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                },
            )
        )
        await agenerate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Write a story"}]}],
            api_key="test-key",
            **output_cap,
        )

    assert json.loads(route.calls.last.request.content)["generationConfig"]["maxOutputTokens"] == 16
