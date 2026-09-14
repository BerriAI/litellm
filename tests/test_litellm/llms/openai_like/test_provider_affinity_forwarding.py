import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai_like import dynamic_config
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig


def _provider(*, responses: bool = False) -> SimpleProviderConfig:
    endpoints = ["/v1/chat/completions"]
    if responses:
        endpoints.append("/v1/responses")
    return SimpleProviderConfig(
        "db_only_provider",
        {
            "base_url": "https://db-only.example/v1",
            "api_key_env": "DYNAMIC_PROVIDER_API_KEY",
            "supported_endpoints": endpoints,
        },
    )


def _chat_response_payload(content: str = "dynamic response") -> dict[str, object]:
    return {
        "id": "chatcmpl-dynamic-provider",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def _responses_payload() -> dict[str, object]:
    return {
        "id": "resp_dynamic_provider",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "test-model",
        "output": [],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        "error": None,
    }


@pytest.fixture(autouse=True)
def _isolate_registry_state():
    original_providers = dict(JSONProviderRegistry._providers)
    dynamic_config._responses_config_cache.clear()
    yield
    JSONProviderRegistry._providers = original_providers
    dynamic_config._responses_config_cache.clear()


def test_dynamic_provider_receives_affinity_header_for_chat():
    from openai import OpenAI

    JSONProviderRegistry._providers = {"db_only_provider": _provider()}
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response_payload())

    client = OpenAI(
        api_key="test-key",
        base_url="https://db-only.example/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    try:
        litellm.completion(
            model="db_only_provider/test-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            client=client,
            extra_headers={"X-Customer-Header": "customer-value"},
            litellm_session_id="session-sync",
            provider_affinity_header="X-Conversation-Id",
        )
    finally:
        client.close()

    assert requests[0].headers["x-conversation-id"] == "session-sync"
    assert requests[0].headers["x-customer-header"] == "customer-value"


def test_dynamic_provider_does_not_use_trace_id_for_chat_affinity():
    from openai import OpenAI

    JSONProviderRegistry._providers = {"db_only_provider": _provider()}
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response_payload())

    client = OpenAI(
        api_key="test-key",
        base_url="https://db-only.example/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    try:
        litellm.completion(
            model="db_only_provider/test-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            client=client,
            metadata={"trace_id": "per-request-trace"},
            provider_affinity_header="X-Conversation-Id",
        )
    finally:
        client.close()

    assert "x-conversation-id" not in requests[0].headers


@pytest.mark.asyncio
async def test_dynamic_provider_receives_affinity_header_for_async_chat():
    from openai import AsyncOpenAI

    JSONProviderRegistry._providers = {"db_only_provider": _provider()}
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response_payload("async response"))

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://db-only.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        response = await litellm.acompletion(
            model="db_only_provider/test-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            client=client,
            litellm_session_id="session-async",
            provider_affinity_header="X-Conversation-Id",
        )
    finally:
        await client.close()

    assert response.choices[0].message.content == "async response"
    assert requests[0].headers["x-conversation-id"] == "session-async"


def test_dynamic_provider_receives_affinity_header_for_streaming_chat():
    from openai import OpenAI

    JSONProviderRegistry._providers = {"db_only_provider": _provider()}
    chunks = [
        {
            "id": "chatcmpl-dynamic-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "streamed"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-dynamic-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    stream_body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    client = OpenAI(
        api_key="test-key",
        base_url="https://db-only.example/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    try:
        response_chunks = list(
            litellm.completion(
                model="db_only_provider/test-model",
                messages=[{"role": "user", "content": "hello"}],
                api_key="test-key",
                client=client,
                stream=True,
                litellm_session_id="session-stream",
                provider_affinity_header="X-Conversation-Id",
            )
        )
    finally:
        client.close()

    assert any(chunk.choices[0].delta.content == "streamed" for chunk in response_chunks)
    assert requests[0].headers["x-conversation-id"] == "session-stream"


def test_dynamic_provider_receives_affinity_header_for_responses():
    JSONProviderRegistry._providers = {"db_only_provider": _provider(responses=True)}
    logging_obj = MagicMock()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_responses_payload())

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        litellm.responses(
            model="db_only_provider/test-model",
            input="hello",
            api_key="test-key",
            litellm_session_id="session-responses",
            provider_affinity_header="X-Conversation-Id",
            litellm_logging_obj=logging_obj,
            client=HTTPHandler(client=http_client),
        )
    finally:
        http_client.close()

    assert requests[0].headers["X-Conversation-Id"] == "session-responses"
    assert (
        logging_obj.update_from_kwargs.call_args.kwargs["litellm_params"]["provider_affinity_header"]
        == "X-Conversation-Id"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_dynamic_provider_receives_affinity_header_for_async_responses(stream: bool):
    JSONProviderRegistry._providers = {"db_only_provider": _provider(responses=True)}
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_responses_payload())

    client = AsyncHTTPHandler()
    await client.close()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        response = await litellm.aresponses(
            model="db_only_provider/test-model",
            input="hello",
            api_key="test-key",
            stream=stream,
            litellm_session_id="session-async-responses",
            provider_affinity_header="X-Conversation-Id",
            client=client,
        )
    finally:
        await client.client.aclose()

    assert requests[0].headers["X-Conversation-Id"] == "session-async-responses"
    if stream:
        assert hasattr(response, "__aiter__")
    else:
        assert getattr(response, "model", None) == "test-model"


def test_builtin_provider_receives_affinity_header():
    from openai import OpenAI

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response_payload())

    client = OpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    try:
        litellm.completion(
            model="openai/test-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            client=client,
            litellm_session_id="session-builtin",
            provider_affinity_header="X-Conversation-Id",
        )
    finally:
        client.close()

    assert requests[0].headers["X-Conversation-Id"] == "session-builtin"
