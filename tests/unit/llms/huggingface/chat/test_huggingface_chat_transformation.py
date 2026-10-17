import json
from collections.abc import Iterator
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.huggingface.common_utils import _fetch_inference_provider_mapping

HUGGINGFACE_MODEL: Final = "meta-llama/Meta-Llama-3-8B-Instruct"
HUGGINGFACE_ROUTER_MODEL: Final = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo"
HUGGINGFACE_COMPLETION_RESPONSE: Final = {
    "id": "hf-completion",
    "object": "chat.completion",
    "created": 11111,
    "model": HUGGINGFACE_ROUTER_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hugging Face response",
                "tool_calls": [],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}
HUGGINGFACE_STREAMING_CHUNKS: Final = (
    {
        "id": "hf-stream-1",
        "object": "chat.completion.chunk",
        "created": 11111,
        "model": HUGGINGFACE_ROUTER_MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hugging "}, "finish_reason": None}],
    },
    {
        "id": "hf-stream-2",
        "object": "chat.completion.chunk",
        "created": 11111,
        "model": HUGGINGFACE_ROUTER_MODEL,
        "choices": [{"index": 0, "delta": {"content": "Face"}, "finish_reason": None}],
    },
    {
        "id": "hf-stream-3",
        "object": "chat.completion.chunk",
        "created": 11111,
        "model": HUGGINGFACE_ROUTER_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
)
HUGGINGFACE_STREAMING_RESPONSE: Final = (
    b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in HUGGINGFACE_STREAMING_CHUNKS)
    + b"data: [DONE]\n\n"
)


class RespxAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, router: respx.MockRouter) -> None:
        self.router: Final = router

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.router.async_handler(request)


def _add_provider_mapping_route(router: respx.MockRouter) -> None:
    route: Final = router.get(f"https://huggingface.co/api/models/{HUGGINGFACE_MODEL}")
    route.return_value = httpx.Response(200, json={"inferenceProviderMapping": PROVIDER_MAPPING_RESPONSE})


def _add_tokenizer_route(router: respx.MockRouter) -> None:
    route: Final = router.head("https://huggingface.co/Xenova/llama-3-tokenizer/resolve/main/tokenizer.json")
    route.return_value = httpx.Response(200)


@pytest.fixture(autouse=True)
def isolate_huggingface_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HF_API_BASE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_BASE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    _fetch_inference_provider_mapping.cache_clear()
    yield
    _fetch_inference_provider_mapping.cache_clear()


PROVIDER_MAPPING_RESPONSE = {
    "fireworks-ai": {
        "status": "live",
        "providerId": "accounts/fireworks/models/llama-v3-8b-instruct",
        "task": "conversational",
    },
    "together": {
        "status": "live",
        "providerId": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "task": "conversational",
    },
    "hf-inference": {
        "status": "live",
        "providerId": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task": "conversational",
    },
}


@pytest.fixture
def mock_provider_mapping() -> Iterator[MagicMock]:
    with patch(
        "litellm.llms.huggingface.chat.transformation.fetch_inference_provider_mapping"
    ) as mock:
        mock.return_value = PROVIDER_MAPPING_RESPONSE
        yield mock


@pytest.mark.usefixtures("fake_provider_credentials")
def test_build_chat_completion_url_function():
    """Test the _build_chat_completion_url helper function"""
    from litellm.llms.huggingface.chat.transformation import (
        _build_chat_completion_url,
    )

    test_cases = [
        ("https://example.com", "https://example.com/v1/chat/completions"),
        ("https://example.com/", "https://example.com/v1/chat/completions"),
        ("https://example.com/v1", "https://example.com/v1/chat/completions"),
        ("https://example.com/v1/", "https://example.com/v1/chat/completions"),
        (
            "https://example.com/v1/chat/completions",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/",
            "https://example.com/custom/path/v1/chat/completions",
        ),
    ]

    for input_url, expected_url in test_cases:
        result = _build_chat_completion_url(input_url)
        assert (
            result == expected_url
        ), f"Failed for input: {input_url}, expected: {expected_url}, got: {result}"


@pytest.mark.parametrize(
    "model, expected_url",
    [
        (
            "meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/v1/chat/completions",
        ),
        (
            "together/meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/together/v1/chat/completions",
        ),
        (
            "novita/meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
        ),
        (
            "http://custom-endpoint.com/v1/chat/completions",
            "http://custom-endpoint.com/v1/chat/completions",
        ),
    ],
)
def test_get_complete_url(model, expected_url):
    """Test that the complete URL is constructed correctly for different providers"""
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    url = config.get_complete_url(
        api_base=None,
        model=model,
        optional_params={},
        stream=False,
        api_key="test_api_key",
        litellm_params={},
    )
    assert url == expected_url


@pytest.mark.parametrize(
    "api_base, model, expected_url",
    [
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "huggingface/tgi",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/v1/chat/completions",
            "huggingface/tgi",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/v1",
            "huggingface/tgi",
            "https://example.com/v1/chat/completions",
        ),
    ],
)
def test_get_complete_url_inference_endpoints(api_base, model, expected_url):
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    url = config.get_complete_url(
        api_base=api_base,
        model=model,
        optional_params={},
        stream=False,
        api_key="test_api_key",
        litellm_params={},
    )
    assert url == expected_url


@pytest.mark.usefixtures("mock_provider_mapping")
@pytest.mark.parametrize(
    "model, expected_model",
    [
        (
            "together/meta-llama/Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        ),
        (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ),
    ],
)
def test_transform_request(model, expected_model):
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    messages = [{"role": "user", "content": "Hello"}]

    transformed_request = config.transform_request(
        model=model,
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert transformed_request["model"] == expected_model
    assert transformed_request["messages"] == messages


@pytest.mark.usefixtures("fake_provider_credentials")
def test_validate_environment():
    """Test that the environment is validated correctly"""
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()

    headers = config.validate_environment(
        headers={},
        model="huggingface/fireworks-ai/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        api_key="test_api_key",
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer test_api_key"
    assert headers["content-type"] == "application/json"


@pytest.mark.respx(assert_all_called=True)
def test_completion_non_streaming_uses_resolved_provider_model(respx_mock: respx.MockRouter) -> None:
    _add_provider_mapping_route(respx_mock)
    route: Final = respx_mock.post("https://router.huggingface.co/together/v1/chat/completions")
    route.return_value = httpx.Response(200, json=HUGGINGFACE_COMPLETION_RESPONSE)
    messages: Final = [{"role": "user", "content": "This is a dummy message"}]

    response: Final = litellm.completion(
        model="huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        api_key="test-api-key",
        client=HTTPHandler(),
    )

    request_body: Final = route.calls[0].request.read()
    assert json.loads(request_body) == {
        "model": HUGGINGFACE_ROUTER_MODEL,
        "messages": messages,
    }
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-api-key"
    assert response.choices[0].message.content == "Hugging Face response"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.model == HUGGINGFACE_ROUTER_MODEL


@pytest.mark.respx(assert_all_called=True)
def test_completion_streaming_uses_resolved_provider_model(respx_mock: respx.MockRouter) -> None:
    _add_provider_mapping_route(respx_mock)
    route: Final = respx_mock.post("https://router.huggingface.co/together/v1/chat/completions")
    route.return_value = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=HUGGINGFACE_STREAMING_RESPONSE,
    )
    messages: Final = [{"role": "user", "content": "This is a dummy message"}]

    chunks: Final = tuple(
        litellm.completion(
            model="huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            stream=True,
            api_key="test-api-key",
            client=HTTPHandler(),
        )
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {"model": HUGGINGFACE_ROUTER_MODEL, "messages": messages, "stream": True}
    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == (
        "Hugging ",
        "Face",
    )
    assert {chunk.model for chunk in chunks} == {"together/meta-llama/Meta-Llama-3-8B-Instruct"}


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_async_completion_non_streaming_uses_resolved_provider_model(
    respx_mock: respx.MockRouter,
) -> None:
    _add_provider_mapping_route(respx_mock)
    route: Final = respx_mock.post("https://router.huggingface.co/together/v1/chat/completions")
    route.return_value = httpx.Response(200, json=HUGGINGFACE_COMPLETION_RESPONSE)
    client: Final = AsyncHTTPHandler(transport=RespxAsyncTransport(respx_mock))
    messages: Final = [{"role": "user", "content": "This is a dummy message"}]

    response: Final = await litellm.acompletion(
        model="huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        api_key="test-api-key",
        client=client,
    )
    await client.client.aclose()

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {"model": HUGGINGFACE_ROUTER_MODEL, "messages": messages}
    assert response.choices[0].message.content == "Hugging Face response"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.model == HUGGINGFACE_ROUTER_MODEL


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_async_completion_streaming_uses_resolved_provider_model(
    respx_mock: respx.MockRouter,
) -> None:
    _add_provider_mapping_route(respx_mock)
    _add_tokenizer_route(respx_mock)
    route: Final = respx_mock.post("https://router.huggingface.co/together/v1/chat/completions")
    route.return_value = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=HUGGINGFACE_STREAMING_RESPONSE,
    )
    client: Final = AsyncHTTPHandler(transport=RespxAsyncTransport(respx_mock))
    messages: Final = [{"role": "user", "content": "This is a dummy message"}]

    stream: Final = await litellm.acompletion(
        model="huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        stream=True,
        api_key="test-api-key",
        client=client,
    )
    chunks: Final = tuple([chunk async for chunk in stream])
    await client.client.aclose()

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {"model": HUGGINGFACE_ROUTER_MODEL, "messages": messages, "stream": True}
    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == (
        "Hugging ",
        "Face",
    )
    assert {chunk.model for chunk in chunks} == {"together/meta-llama/Meta-Llama-3-8B-Instruct"}


@pytest.mark.respx(assert_all_called=True)
def test_completion_with_api_base_uses_endpoint_url(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://custom.huggingface.test/v1/chat/completions")
    route.return_value = httpx.Response(200, json=HUGGINGFACE_COMPLETION_RESPONSE)
    messages: Final = [{"role": "user", "content": "This is a test message"}]

    response: Final = litellm.completion(
        model="huggingface/tgi",
        messages=messages,
        api_base="https://custom.huggingface.test",
        api_key="test-api-key",
        client=HTTPHandler(),
    )

    assert str(route.calls[0].request.url) == "https://custom.huggingface.test/v1/chat/completions"
    assert response.choices[0].message.content == "Hugging Face response"


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_async_completion_with_api_base_uses_endpoint_url(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("https://custom.huggingface.test/v1/chat/completions")
    route.return_value = httpx.Response(200, json=HUGGINGFACE_COMPLETION_RESPONSE)
    client: Final = AsyncHTTPHandler(transport=RespxAsyncTransport(respx_mock))
    messages: Final = [{"role": "user", "content": "This is a test message"}]

    response: Final = await litellm.acompletion(
        model="huggingface/tgi",
        messages=messages,
        api_base="https://custom.huggingface.test",
        api_key="test-api-key",
        client=client,
    )
    await client.client.aclose()

    assert str(route.calls[0].request.url) == "https://custom.huggingface.test/v1/chat/completions"
    assert response.choices[0].message.content == "Hugging Face response"


@pytest.mark.respx(assert_all_called=True)
def test_completion_streaming_with_api_base_uses_endpoint_url(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://custom.huggingface.test/v1/chat/completions")
    route.return_value = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=HUGGINGFACE_STREAMING_RESPONSE,
    )
    messages: Final = [{"role": "user", "content": "This is a test message"}]

    chunks: Final = tuple(
        litellm.completion(
            model="huggingface/tgi",
            messages=messages,
            api_base="https://custom.huggingface.test",
            stream=True,
            api_key="test-api-key",
            client=HTTPHandler(),
        )
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["stream"] is True
    assert request_body["messages"] == messages
    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == (
        "Hugging ",
        "Face",
    )


@pytest.mark.respx(assert_all_called=True)
def test_tool_call_without_arguments_is_preserved(respx_mock: respx.MockRouter) -> None:
    _add_provider_mapping_route(respx_mock)
    tool_response: Final = {
        **HUGGINGFACE_COMPLETION_RESPONSE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_faq",
                            "type": "function",
                            "function": {"name": "Get-FAQ", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    route: Final = respx_mock.post("https://router.huggingface.co/together/v1/chat/completions")
    route.return_value = httpx.Response(200, json=tool_response)
    messages: Final = [{"role": "user", "content": "Get the FAQ"}]
    tools: Final = [
        {
            "type": "function",
            "function": {"name": "Get-FAQ", "description": "Get FAQ information", "parameters": {"type": "object"}},
        }
    ]

    response: Final = litellm.completion(
        model="huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        api_key="test-api-key",
        client=HTTPHandler(),
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["model"] == HUGGINGFACE_ROUTER_MODEL
    assert request_body["tools"] == tools
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "Get-FAQ"
    assert response.choices[0].message.tool_calls[0].function.arguments == ""
