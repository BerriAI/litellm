import asyncio
import json
import os
import sys
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

import litellm

pytestmark = pytest.mark.flaky(condition=False)


def _initialize_proxy_with_config(config: dict, tmp_path) -> TestClient:
    """
    Initialize the proxy server with a temporary config file and return a TestClient.

    IMPORTANT: proxy_server.initialize() mutates module-level globals. We must call
    cleanup_router_config_variables() before initializing to prevent cross-test bleed.
    """
    from litellm.proxy.proxy_server import app, cleanup_router_config_variables, initialize

    cleanup_router_config_variables()

    config_fp = tmp_path / "proxy_config.yaml"
    config_fp.write_text(yaml.safe_dump(config))

    asyncio.run(initialize(config=str(config_fp), debug=True))
    return TestClient(app)


def _make_minimal_chat_completion_response(model: str) -> litellm.ModelResponse:
    response = litellm.ModelResponse()
    response.model = model
    response.choices[0].message.content = "hello"  # type: ignore[union-attr]
    response.choices[0].finish_reason = "stop"  # type: ignore[union-attr]
    return response


def _make_model_response_stream_chunk(model: str) -> litellm.ModelResponseStream:
    """
    Create a minimal OpenAI-compatible chat.completion.chunk object.
    """
    chunk_dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
                "finish_reason": None,
            }
        ],
    }
    return litellm.ModelResponseStream(**chunk_dict)


def test_proxy_chat_completion_does_not_return_provider_prefixed_model(tmp_path, monkeypatch):
    """
    Regression test:

    - Client asks for `model="vllm-model"` (no provider prefix)
    - Internal provider path uses `hosted_vllm/...`
    - Proxy should not leak `hosted_vllm/` in the client-facing `model` field.
    """
    client_model = "vllm-model"
    internal_model = f"hosted_vllm/{client_model}"

    client = _initialize_proxy_with_config(
        config={
            "general_settings": {"master_key": "sk-1234"},
            "model_list": [
                {
                    "model_name": client_model,
                    "litellm_params": {"model": internal_model},
                }
            ],
        },
        tmp_path=tmp_path,
    )

    # Patch router call to avoid making any real network request.
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server.llm_router,  # type: ignore[arg-type]
        "acompletion",
        AsyncMock(return_value=_make_minimal_chat_completion_response(model=internal_model)),
    )

    # Also no-op proxy logging hooks to keep this test focused and deterministic.
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "during_call_hook", AsyncMock(return_value=None))
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "update_request_status", AsyncMock(return_value=None))
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_success_hook", AsyncMock(side_effect=lambda **kwargs: kwargs["response"]))

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-1234"},
        json={"model": client_model, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == client_model
    assert not body["model"].startswith("hosted_vllm/")


@pytest.mark.asyncio
async def test_proxy_streaming_chunks_do_not_return_provider_prefixed_model(monkeypatch):
    """
    Regression test for streaming:

    Even if a streaming chunk contains `model="hosted_vllm/<...>"`, the proxy SSE layer
    should not leak the provider prefix to the client.
    """
    client_model = "vllm-model"
    internal_model = f"hosted_vllm/{client_model}"

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy import proxy_server

    # Patch proxy_logging_obj hooks so async_data_generator yields exactly our chunk.
    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=internal_model)

    monkeypatch.setattr(proxy_server.proxy_logging_obj, "async_post_call_streaming_iterator_hook", _iterator_hook)
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-1234")

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={"model": client_model},
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    # First chunk is expected to be JSON, last chunk is [DONE]
    assert len(chunks) >= 2
    first = chunks[0]
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    assert payload["model"] == client_model
    assert not payload["model"].startswith("hosted_vllm/")


@pytest.mark.asyncio
async def test_proxy_streaming_chunks_use_client_requested_model_before_alias_mapping(monkeypatch):
    """
    Regression test for alias mapping on streaming:

    - `common_processing_pre_call_logic` can rewrite `request_data["model"]` via model_alias_map / key-specific aliases.
    - Non-streaming responses are restamped using the original client-requested model (captured before the rewrite).
    - Streaming chunks must do the same to avoid mismatched `model` values between streaming and non-streaming.
    """
    client_model_alias = "alias-model"
    canonical_model = "vllm-model"
    internal_model = f"hosted_vllm/{canonical_model}"

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy import proxy_server

    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=internal_model)

    monkeypatch.setattr(proxy_server.proxy_logging_obj, "async_post_call_streaming_iterator_hook", _iterator_hook)
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-1234")

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={
            "model": canonical_model,
            "_litellm_client_requested_model": client_model_alias,
        },
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    assert len(chunks) >= 2
    first = chunks[0]
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    assert payload["model"] == client_model_alias
    assert not payload["model"].startswith("hosted_vllm/")
