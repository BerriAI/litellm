import json
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.llmman.chat.transformation import LlmmanChatConfig

DEFAULT_API_BASE: Final = "http://127.0.0.1:17434/v1"
ENV: Final = {
    "LLMMAN_API_BASE": "https://env-api.example.com",
    "LLMMAN_API_KEY": "env-key",
}


@pytest.mark.parametrize(
    "api_base, api_key, env, expected_base, expected_key",
    [
        ("https://user-api.example.com", "user-key", ENV, "https://user-api.example.com", "user-key"),
        (None, None, ENV, "https://env-api.example.com", "env-key"),
        (None, None, {}, DEFAULT_API_BASE, "fake-api-key"),
        ("", "", ENV, "https://env-api.example.com", "env-key"),
        ("", "", {}, DEFAULT_API_BASE, "fake-api-key"),
        ("https://user-api.example.com", None, ENV, "https://user-api.example.com", "env-key"),
        (None, "user-key", ENV, "https://env-api.example.com", "user-key"),
    ],
)
def test_get_openai_compatible_provider_info(api_base, api_key, env, expected_base, expected_key):
    with patch.dict("os.environ", env, clear=True):
        assert LlmmanChatConfig().get_openai_compatible_provider_info(api_base, api_key) == (
            expected_base,
            expected_key,
        )


def test_get_llm_provider_routes_llmman_prefix():
    with patch.dict("os.environ", {}, clear=True):
        result = litellm.get_llm_provider("llmman/my-custom-test-model")

    assert result == ("my-custom-test-model", "llmman", "fake-api-key", DEFAULT_API_BASE)


@respx.mock
def test_completion_hits_default_llmman_endpoint():
    route = respx.post(f"{DEFAULT_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "my-custom-test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    with patch.dict("os.environ", {}, clear=True):
        response = litellm.completion(
            model="llmman/my-custom-test-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            max_tokens=100,
        )

    assert response.choices[0].message.content == "hi"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-api-key"
    body = json.loads(request.content)
    assert body["model"] == "my-custom-test-model"
    assert body["max_tokens"] == 100


@respx.mock
def test_embedding_hits_default_llmman_endpoint():
    route = respx.post(f"{DEFAULT_API_BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "my-embedding-model",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )

    with patch.dict("os.environ", {}, clear=True):
        response = litellm.embedding(model="llmman/my-embedding-model", input=["hello"])

    assert response.data[0]["embedding"] == [0.1, 0.2]
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-api-key"
    assert json.loads(request.content)["model"] == "my-embedding-model"
