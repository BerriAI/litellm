import json
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_COMPLETION: Final = {
    "id": "chatcmpl_test",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "qwen/qwen3.8-27b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


@pytest.mark.parametrize(
    "model,custom_provider,api_base",
    (
        ("iteracompute/qwen/qwen3.8-27b", None, None),
        ("qwen/qwen3.8-27b", "iteracompute", None),
        ("qwen/qwen3.8-27b", None, "https://api.iteracompute.com/v1"),
    ),
)
def test_provider_resolution_preserves_model_namespace(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    custom_provider: str | None,
    api_base: str | None,
) -> None:
    monkeypatch.setenv("ITERACOMPUTE_API_KEY", "test-key")
    monkeypatch.delenv("ITERACOMPUTE_API_BASE", raising=False)

    assert get_llm_provider(
        model=model,
        custom_llm_provider=custom_provider,
        api_base=api_base,
        api_key=None,
    ) == (
        "qwen/qwen3.8-27b",
        "iteracompute",
        "test-key",
        "https://api.iteracompute.com/v1",
    )


@pytest.mark.parametrize("explicit", (False, True))
def test_credentials_respect_explicit_arguments_before_environment(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.setenv("ITERACOMPUTE_API_KEY", "env-key")
    monkeypatch.setenv("ITERACOMPUTE_API_BASE", "https://env.example/v1")

    assert get_llm_provider(
        model="iteracompute/qwen/qwen3.8-27b",
        custom_llm_provider=None,
        api_base="https://explicit.example/v1" if explicit else None,
        api_key="explicit-key" if explicit else None,
    ) == (
        "qwen/qwen3.8-27b",
        "iteracompute",
        "explicit-key" if explicit else "env-key",
        "https://explicit.example/v1" if explicit else "https://env.example/v1",
    )


def test_completion_sends_model_and_bearer_key_without_provider_prefix() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.iteracompute.com/v1/chat/completions"
        ).respond(200, json=_COMPLETION)
        response: Final = litellm.completion(
            model="iteracompute/qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            max_tokens=512,
        )

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512,
    }
    assert response.choices[0].message.content == "Hello"
    assert response.usage.total_tokens == 7


def test_streaming_returns_text_and_usage() -> None:
    event: Final = {
        "id": "chatcmpl_test",
        "object": "chat.completion.chunk",
        "created": 1_790_000_000,
        "model": "qwen/qwen3.8-27b",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.iteracompute.com/v1/chat/completions"
        ).respond(
            200,
            text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
        chunks: Final = tuple(
            litellm.completion(
                model="iteracompute/qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="test-key",
                stream=True,
                stream_options={"include_usage": True},
            )
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen/qwen3.8-27b"
    assert body["stream_options"] == {"include_usage": True}
    assert (
        "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        )
        == "Hello"
    )
    assert (
        next(
            chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None
        ).total_tokens
        == 7
    )


def test_invalid_key_surfaces_authentication_error() -> None:
    with respx.mock() as upstream:
        upstream.post("https://api.iteracompute.com/v1/chat/completions").respond(
            401,
            json={
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            },
        )
        with pytest.raises(litellm.AuthenticationError, match="Invalid API key"):
            litellm.completion(
                model="iteracompute/qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="invalid-test-key",
            )


def test_responses_bridge_keeps_model_namespace() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.iteracompute.com/v1/chat/completions"
        ).respond(200, json=_COMPLETION)
        response: Final = litellm.responses(
            model="iteracompute/qwen/qwen3.8-27b", input="Hello", api_key="test-key"
        )

    assert json.loads(route.calls.last.request.content)["model"] == "qwen/qwen3.8-27b"
    assert response.output[0].content[0].text == "Hello"


@pytest.mark.asyncio
async def test_messages_bridge_keeps_model_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.iteracompute.com/v1/chat/completions"
        ).respond(200, json=_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="iteracompute/qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=512,
            api_key="test-key",
        )

    assert json.loads(route.calls.last.request.content)["model"] == "qwen/qwen3.8-27b"
    assert response["content"][0]["text"] == "Hello"
