import json
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [("foundry_local/local-model:1", None), ("local-model:1", "foundry_local")],
)
@pytest.mark.parametrize(
    ("api_base", "api_key", "env_base", "env_key", "expected_base", "expected_key"),
    [
        (None, None, None, None, None, "fake-api-key"),
        (None, None, "http://127.0.0.1:61234/v1", None, "http://127.0.0.1:61234/v1", "fake-api-key"),
        (None, None, "http://127.0.0.1:61234/v1", "env-key", "http://127.0.0.1:61234/v1", "env-key"),
        ("http://localhost:5272/v1", None, None, "env-key", "http://localhost:5272/v1", "env-key"),
        (None, "explicit-key", "http://127.0.0.1:61234/v1", None, "http://127.0.0.1:61234/v1", "explicit-key"),
        ("http://localhost:5272/v1", "explicit-key", None, None, "http://localhost:5272/v1", "explicit-key"),
        (
            "http://localhost:5272/v1",
            "explicit-key",
            "http://127.0.0.1:61234/v1",
            "env-key",
            "http://localhost:5272/v1",
            "explicit-key",
        ),
    ],
)
def test_should_resolve_foundry_endpoint_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    custom_llm_provider: str | None,
    api_base: str | None,
    api_key: str | None,
    env_base: str | None,
    env_key: str | None,
    expected_base: str | None,
    expected_key: str,
) -> None:
    monkeypatch.delenv("FOUNDRY_LOCAL_API_BASE", raising=False)
    monkeypatch.delenv("FOUNDRY_LOCAL_API_KEY", raising=False)
    if env_base is not None:
        monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", env_base)
    if env_key is not None:
        monkeypatch.setenv("FOUNDRY_LOCAL_API_KEY", env_key)

    assert get_llm_provider(model, custom_llm_provider=custom_llm_provider, api_base=api_base, api_key=api_key) == (
        "local-model:1",
        "foundry_local",
        expected_key,
        expected_base,
    )


def _respond(request: httpx.Request, stream: bool) -> httpx.Response:
    assert str(request.url) == "http://127.0.0.1:61234/v1/chat/completions"
    payload: Final = json.loads(request.content)
    assert payload["model"] == "local-model:1"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 16
    assert bool(payload.get("stream")) is stream
    if stream:
        chunk: Final = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "local-model:1",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        }
        final_chunk: Final = {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n",
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": "local-model:1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


@pytest.mark.parametrize("stream", [False, True])
def test_should_complete_chat_over_foundry_http_service(stream: bool) -> None:
    with (
        httpx.Client(transport=httpx.MockTransport(lambda request: _respond(request, stream))) as http_client,
        OpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = litellm.completion(
            model="foundry_local/local-model:1",
            api_base="http://127.0.0.1:61234/v1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            stream=stream,
            client=client,
        )
        if stream:
            assert not isinstance(response, ModelResponse)
            chunks: Final = tuple(response)
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.total_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_should_complete_chat_asynchronously_over_foundry_http_service(stream: bool) -> None:
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: _respond(request, stream))) as http_client,
        AsyncOpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = await litellm.acompletion(
            model="foundry_local/local-model:1",
            api_base="http://127.0.0.1:61234/v1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            stream=stream,
            client=client,
        )
        if stream:
            assert not isinstance(response, ModelResponse)
            chunks: Final = [chunk async for chunk in response]
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.total_tokens == 3
