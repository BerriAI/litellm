import asyncio
import importlib
import json
from collections.abc import Mapping
from types import ModuleType
from typing import Final, Protocol
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai._types import Response as SDKResponse

import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.utils import ImageResponse


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.openai.com/v1",
        "https://api.openai.com:443/v1",
        "https://southcentralus.privatelink.api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        "https://us.api.openai.com/v1",
        "HTTPS://API.OPENAI.COM/v1/",
    ],
)
def test_get_stream_options_defaults_include_usage_on_every_openai_backed_host(api_base):
    """
    PrivateLink and regional hostnames reach the real OpenAI backend, so a stream with no caller
    stream_options must ask for the usage chunk exactly as the default base does. Regression guard
    for LIT-6875: spend for those deployments fell back to local token counting.
    """
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {
        "stream_options": {"include_usage": True}
    }


@pytest.mark.parametrize(
    "api_base",
    [
        "https://my-gateway.example/v1",
        "https://api.openai.com.evil.example/v1",
        "https://notapi.openai.com/v1",
        "https://gateway.example/v1?upstream=api.openai.com",
        "https://openai.internal.example/api.openai.com/v1",
    ],
)
def test_get_stream_options_leaves_foreign_hosts_without_a_usage_default(api_base):
    """Only the host decides: an OpenAI-compatible backend elsewhere may not support stream_options at all."""
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {}


@pytest.mark.parametrize(
    "api_base",
    ["https://southcentralus.privatelink.api.openai.com/v1", "https://my-gateway.example/v1"],
)
def test_get_stream_options_passes_caller_stream_options_through_on_any_host(api_base):
    caller_options = {"include_usage": False}
    assert OpenAIChatCompletion().get_stream_options(stream_options=caller_options, api_base=api_base) == {
        "stream_options": caller_options
    }


@pytest.fixture(params=("httpx", "sdk"))
def http_backend(request: pytest.FixtureRequest) -> ModuleType:
    return importlib.import_module("httpx" if request.param == "httpx" else SDKResponse.__module__)


class _Request(Protocol):
    @property
    def content(self) -> bytes: ...

    @property
    def extensions(self) -> Mapping[str, object]: ...


@pytest.mark.asyncio
async def test_acompletion_returns_json_reply_over_injected_transport(http_backend: ModuleType):
    outbound: Final = asyncio.Queue()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        assert request.extensions["timeout"] == {"connect": 1, "read": None, "write": 2, "pool": 3}
        outbound.put_nowait(json.loads(request.content))
        return http_backend.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "smoke-json-reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        response: Final = await asyncio.wait_for(
            litellm.acompletion(
                model="openai/gpt-5.6",
                api_key="transport-only",
                client=client,
                messages=[{"role": "user", "content": "smoke-json-request"}],
                timeout=http_backend.Timeout(connect=1, read=None, write=2, pool=3),
                num_retries=0,
                max_retries=0,
            ),
            timeout=10,
        )
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["model"] == "gpt-5.6"
        assert request["messages"] == [{"role": "user", "content": "smoke-json-request"}]
        assert not request.get("stream")
        assert outbound.empty()
        assert response.choices[0].message.content == "smoke-json-reply"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 15
        assert not http_client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_error"),
    ((400, litellm.BadRequestError), (429, litellm.RateLimitError), (503, litellm.ServiceUnavailableError)),
)
async def test_acompletion_preserves_public_errors_for_both_http_clients(
    http_backend: ModuleType, status: int, expected_error: type[Exception]
) -> None:
    def respond(request: _Request) -> httpx.Response | SDKResponse:
        return http_backend.Response(status, json={"error": {"message": "upstream failure", "type": "api_error"}})

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        sdk_client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client, max_retries=0)
        with pytest.raises(expected_error, match="upstream failure"):
            await litellm.acompletion(
                model="openai/gpt-5.6",
                client=sdk_client,
                messages=[{"role": "user", "content": "request"}],
                num_retries=0,
                max_retries=0,
            )
        assert not http_client.is_closed


@pytest.mark.asyncio
async def test_acompletion_maps_both_transport_timeouts_to_litellm_timeout(http_backend: ModuleType) -> None:
    def respond(request: _Request) -> httpx.Response | SDKResponse:
        raise http_backend.ReadTimeout("upstream timeout", request=request)

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        sdk_client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client, max_retries=0)
        with pytest.raises(litellm.Timeout):
            await litellm.acompletion(
                model="openai/gpt-5.6",
                client=sdk_client,
                messages=[{"role": "user", "content": "request"}],
                num_retries=0,
                max_retries=0,
            )
        assert not http_client.is_closed


@pytest.mark.asyncio
async def test_acompletion_streams_text_deltas_over_injected_transport(http_backend: ModuleType):
    outbound: Final = asyncio.Queue()

    def chunk(delta: dict, finish: str | None) -> bytes:
        body: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        outbound.put_nowait(json.loads(request.content))
        usage: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        content: Final = b"".join(
            (
                chunk({"role": "assistant", "content": "Hel"}, None),
                chunk({"content": "lo"}, "stop"),
                f"data: {json.dumps(usage)}\n\n".encode(),
                b"data: [DONE]\n\n",
            )
        )
        return http_backend.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        stream: Final = await litellm.acompletion(
            model="openai/gpt-5.6",
            api_key="transport-only",
            client=client,
            messages=[{"role": "user", "content": "smoke-stream-request"}],
            stream=True,
            num_retries=0,
            max_retries=0,
        )
        chunks: Final = []

        async def drain() -> None:
            async for part in stream:
                chunks.append(part)

        await asyncio.wait_for(drain(), timeout=10)
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["stream"] is True
        assert outbound.empty()
        assert (
            "".join(part.choices[0].delta.content or "" for part in chunks if part.choices and part.choices[0].delta)
            == "Hello"
        )
        last_finish: Final = next(
            part.choices[0].finish_reason for part in reversed(chunks) if part.choices and part.choices[0].finish_reason
        )
        assert last_finish == "stop"


@pytest.mark.asyncio
async def test_acompletion_streams_tool_call_arguments_over_injected_transport(http_backend: ModuleType):
    outbound: Final = asyncio.Queue()
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    def chunk(delta: dict, finish: str | None) -> bytes:
        body: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        outbound.put_nowait(json.loads(request.content))
        content: Final = b"".join(
            (
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    None,
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}, None),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}, "tool_calls"),
                b"data: [DONE]\n\n",
            )
        )
        return http_backend.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        messages: Final = [{"role": "user", "content": "weather in Paris"}]
        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o",
            api_key="transport-only",
            client=client,
            messages=messages,
            tools=tools,
            stream=True,
            num_retries=0,
            max_retries=0,
        )
        chunks: Final = []

        async def drain() -> None:
            async for part in stream:
                chunks.append(part)

        await asyncio.wait_for(drain(), timeout=10)
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["stream"] is True
        assert request["tools"][0]["function"]["name"] == "get_weather"
        assert outbound.empty()
        rebuilt: Final = litellm.stream_chunk_builder(chunks, messages=messages)
        tool_call: Final = rebuilt.choices[0].message.tool_calls[0]
        assert tool_call.id == "call-1"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"city": "Paris"}
        assert rebuilt.choices[0].finish_reason == "tool_calls"


_PROVIDER_HEADERS: Final = {"x-request-id": "req_openai", "x-ratelimit-remaining-requests": "41"}


def _image_generation_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"created": 1, "data": [{"b64_json": "abc"}]}, headers=_PROVIDER_HEADERS
        )
    )


def _speech_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"audio-bytes", headers={**_PROVIDER_HEADERS, "content-type": "audio/mpeg"}
        )
    )


def _assert_provider_headers_recorded(response) -> None:
    assert response._hidden_params["headers"]["x-request-id"] == "req_openai"
    assert response._hidden_params["additional_headers"]["llm_provider-x-request-id"] == "req_openai"
    assert response._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == "41"


def _image_generation_kwargs() -> dict:
    return {
        "model": "gpt-image-2",
        "prompt": "a cat",
        "timeout": 10,
        "optional_params": {},
        "logging_obj": Mock(),
        "api_key": "transport-only",
        "model_response": ImageResponse(),
    }


def test_image_generation_records_provider_response_headers():
    with httpx.Client(transport=_image_generation_transport()) as http_client:
        response = OpenAIChatCompletion().image_generation(
            client=OpenAI(api_key="transport-only", http_client=http_client), **_image_generation_kwargs()
        )

    _assert_provider_headers_recorded(response)


@pytest.mark.asyncio
async def test_aimage_generation_records_provider_response_headers():
    async with httpx.AsyncClient(transport=_image_generation_transport()) as http_client:
        response = await OpenAIChatCompletion().image_generation(
            client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
            aimg_generation=True,
            **_image_generation_kwargs(),
        )

    _assert_provider_headers_recorded(response)


def _audio_speech_kwargs() -> dict:
    return {
        "model": "gpt-4o-mini-tts",
        "input": "hello",
        "voice": "alloy",
        "optional_params": {},
        "api_key": "transport-only",
        "api_base": None,
        "organization": None,
        "project": None,
        "max_retries": 0,
        "timeout": 10,
        "logging_obj": Mock(),
    }


def test_audio_speech_records_provider_response_headers():
    with httpx.Client(transport=_speech_transport()) as http_client:
        response = OpenAIChatCompletion().audio_speech(
            client=OpenAI(api_key="transport-only", http_client=http_client), **_audio_speech_kwargs()
        )

    _assert_provider_headers_recorded(response)


@pytest.mark.asyncio
async def test_async_audio_speech_records_provider_response_headers():
    async with httpx.AsyncClient(transport=_speech_transport()) as http_client:
        response = await OpenAIChatCompletion().audio_speech(
            client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
            aspeech=True,
            **_audio_speech_kwargs(),
        )

    _assert_provider_headers_recorded(response)
