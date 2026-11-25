import json
from typing import Final

import pytest
import respx

import litellm

WEFLAYR_API_BASE: Final = "https://gateway.weflayr.com/v1"


def test_weflayr_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WEFLAYR_API_KEY", "weflayr-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="weflayr/gpt-5-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-5-mini"
    assert provider == "weflayr"
    assert api_key == "weflayr-test-key"
    assert api_base == WEFLAYR_API_BASE


def test_weflayr_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WEFLAYR_API_KEY", "weflayr-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="weflayr/gpt-5-mini",
        custom_llm_provider=None,
        api_base="https://gateway.internal.example/v1",
        api_key="weflayr-explicit-key",
    )

    assert provider == "weflayr"
    assert api_key == "weflayr-explicit-key"
    assert api_base == "https://gateway.internal.example/v1"


def test_weflayr_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WEFLAYR_API_KEY", "weflayr-test-key")
    monkeypatch.setenv("WEFLAYR_API_BASE", "https://gateway.staging.example/v1")

    _, provider, _, api_base = get_llm_provider(model="weflayr/gpt-5-mini")

    assert provider == "weflayr"
    assert api_base == "https://gateway.staging.example/v1"


@pytest.mark.parametrize("model", ["gpt-5-mini", "claude-sonnet-4-5"])
def test_weflayr_api_base_routes_unprefixed_model_to_weflayr(model: str):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    resolved_model, provider, api_key, api_base = get_llm_provider(
        model=model,
        api_base=WEFLAYR_API_BASE,
        api_key="weflayr-test-key",
    )

    assert resolved_model == model
    assert provider == "weflayr"
    assert api_key == "weflayr-test-key"
    assert api_base == WEFLAYR_API_BASE


CHAT_RESPONSE: Final = {
    "id": "chatcmpl_weflayr",
    "object": "chat.completion",
    "created": 1_789_550_000,
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Weflayr"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_weflayr_chat_completions_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{WEFLAYR_API_BASE}/chat/completions").respond(200, json=CHAT_RESPONSE)
        response: Final = litellm.completion(
            model="weflayr/gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="weflayr-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer weflayr-test-key"
    assert body["model"] == "gpt-5-mini"
    assert response.choices[0].message.content == "Hello from Weflayr"


def test_weflayr_routing_strategy_model_and_call_metadata_reach_the_gateway():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{WEFLAYR_API_BASE}/chat/completions").respond(200, json=CHAT_RESPONSE)
        litellm.completion(
            model="weflayr/strategy/production-default",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="weflayr-test-key",
            extra_body={"weflayr": {"call_name": "greeting", "customer_name": "acme"}},
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "strategy/production-default"
    assert body["weflayr"] == {"call_name": "greeting", "customer_name": "acme"}


def test_weflayr_chat_completions_streaming_request():
    chunks = [
        'data: {"id":"chatcmpl_weflayr","object":"chat.completion.chunk","created":1789550000,"model":"gpt-5-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_weflayr","object":"chat.completion.chunk","created":1789550000,"model":"gpt-5-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{WEFLAYR_API_BASE}/chat/completions").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )
        response: Final = litellm.completion(
            model="weflayr/gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            api_key="weflayr-test-key",
        )
        collected = [chunk for chunk in response]

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["stream"] is True
    assert any(chunk.choices and chunk.choices[0].delta.content == "Hi" for chunk in collected)


def test_weflayr_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{WEFLAYR_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_weflayr",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "gpt-5-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_weflayr",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Weflayr", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="weflayr/gpt-5-mini",
            input="Say hello",
            api_key="weflayr-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer weflayr-test-key"
    assert body["model"] == "gpt-5-mini"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Weflayr"


def test_weflayr_embeddings_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{WEFLAYR_API_BASE}/embeddings").respond(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
        response: Final = litellm.embedding(
            model="weflayr/text-embedding-3-small",
            input=["Say hello"],
            api_key="weflayr-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer weflayr-test-key"
    assert body["model"] == "text-embedding-3-small"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
