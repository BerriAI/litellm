import asyncio
import json
import os
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm

CONFIG_DIR: Final = Path(__file__).resolve().parent / "test_configs"
CUSTOM_CALLBACKS_PATH: Final = CONFIG_DIR / "custom_callbacks.py"
OPENAI_BASE: Final = "https://api.openai.com/v1"

CHAT_COMPLETION_BODY: Final = {
    "id": "chatcmpl-unit-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1-nano",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "litellm poem"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
}

EMBEDDING_BODY: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}

STREAM_CHUNK_1: Final = {
    "id": "chatcmpl-unit-stream",
    "object": "chat.completion.chunk",
    "created": 1700000000,
    "model": "gpt-4.1-nano",
    "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": "Lite"}, "finish_reason": None}
    ],
}
STREAM_CHUNK_2: Final = {
    "id": "chatcmpl-unit-stream",
    "object": "chat.completion.chunk",
    "created": 1700000000,
    "model": "gpt-4.1-nano",
    "choices": [{"index": 0, "delta": {"content": "LLM"}, "finish_reason": "stop"}],
}
STREAM_BODY: Final = (
    f"data: {json.dumps(STREAM_CHUNK_1)}\n\n"
    f"data: {json.dumps(STREAM_CHUNK_2)}\n\n"
    "data: [DONE]\n\n"
).encode()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> TestClient:
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
    )

    monkeypatch.setenv("LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY", "true")
    cleanup_router_config_variables()
    asyncio.run(initialize(config=str(CONFIG_DIR / "test_custom_logger.yaml")))
    return TestClient(app)


def _custom_logger() -> object:
    from litellm.proxy.types_utils.utils import get_instance_fn

    return get_instance_fn(
        value="custom_callbacks.my_custom_logger",
        config_file_path=str(CUSTOM_CALLBACKS_PATH),
    )


def test_chat_completion(client: TestClient, respx_mock: respx.MockRouter) -> None:
    upstream: Final = respx_mock.post(f"{OPENAI_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION_BODY)
    )
    my_custom_logger = _custom_logger()
    litellm.callbacks = [my_custom_logger]

    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "Azure OpenAI GPT-4 Canada",
            "messages": [{"role": "user", "content": "write a litellm poem"}],
            "max_tokens": 10,
        },
    )

    assert response.status_code == 200, response.text
    assert my_custom_logger.done_event.wait(10)
    assert my_custom_logger.async_success is True
    assert my_custom_logger.async_completion_kwargs["model"] == "gpt-4.1-nano"
    litellm_params: Final = my_custom_logger.async_completion_kwargs["litellm_params"]

    metadata: Final = litellm_params.get("metadata")
    assert "user_api_key" in metadata
    assert "user_api_key_metadata" in metadata
    assert "headers" in metadata

    model_info: Final = litellm_params.get("model_info")
    assert model_info.get("id") == "gm"
    assert model_info.get("mode") == "chat"
    assert model_info.get("input_cost_per_token") == 0.0002

    proxy_server_request: Final = litellm_params.get("proxy_server_request")
    assert proxy_server_request.get("url") == "http://testserver/chat/completions"
    assert proxy_server_request.get("method") == "POST"
    assert "authorization" not in proxy_server_request["headers"]
    body: Final = proxy_server_request["body"]
    assert body.get("model") == "Azure OpenAI GPT-4 Canada"
    assert body.get("messages") == [
        {"role": "user", "content": "write a litellm poem"}
    ]
    assert body.get("max_tokens") == 10
    assert upstream.call_count == 1


def test_chat_completion_stream(client: TestClient, respx_mock: respx.MockRouter) -> None:
    upstream: Final = respx_mock.post(f"{OPENAI_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, content=STREAM_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    my_custom_logger = _custom_logger()
    litellm.callbacks = [my_custom_logger]
    assert my_custom_logger.streaming_response_obj is None

    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "Azure OpenAI GPT-4 Canada",
            "messages": [{"role": "user", "content": "write 1 line poem about LiteLLM"}],
            "max_tokens": 40,
            "stream": True,
        },
    )

    assert response.status_code == 200, response.text
    complete_response: Final = "".join(
        json.loads(data)["choices"][0]["delta"].get("content") or ""
        for data in (
            line.replace("data: ", "") for line in response.iter_lines() if line
        )
        if "[DONE]" not in data
    )

    assert complete_response == "LiteLLM"
    assert my_custom_logger.done_event.wait(10)
    assert (
        complete_response
        == my_custom_logger.streaming_response_obj["choices"][0]["message"]["content"]
    )
    assert upstream.call_count == 1


def test_embedding(client: TestClient, respx_mock: respx.MockRouter) -> None:
    upstream: Final = respx_mock.post(f"{OPENAI_BASE}/embeddings").mock(
        return_value=httpx.Response(200, json=EMBEDDING_BODY)
    )
    my_custom_logger = _custom_logger()
    litellm.callbacks = [my_custom_logger]
    assert my_custom_logger.async_success_embedding is False

    response: Final = client.post(
        "/embeddings",
        json={"model": "azure-embedding-model", "input": ["hello"]},
    )

    assert response.status_code == 200, response.text
    assert my_custom_logger.done_event.wait(10)
    assert my_custom_logger.async_success_embedding is True
    assert my_custom_logger.async_embedding_kwargs["model"] == "text-embedding-ada-002"
    litellm_params: Final = my_custom_logger.async_embedding_kwargs["litellm_params"]

    metadata: Final = litellm_params.get("metadata")
    assert "user_api_key" in metadata
    assert "headers" in metadata

    proxy_server_request: Final = litellm_params.get("proxy_server_request")
    assert proxy_server_request.get("url") == "http://testserver/embeddings"
    assert proxy_server_request.get("method") == "POST"
    body: Final = proxy_server_request["body"]
    assert body.get("model") == "azure-embedding-model"
    assert body.get("input") == ["hello"]

    model_info: Final = litellm_params.get("model_info")
    assert model_info.get("mode") == "embedding"
    assert model_info.get("id") == "hello"
    assert model_info.get("input_cost_per_token") == 0.002
    assert upstream.call_count == 1
