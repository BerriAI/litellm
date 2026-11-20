import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_kaer_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("KAER_API_KEY", "kaer-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="kaer/janus-1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "janus-1-flash"
    assert provider == "kaer"
    assert api_key == "kaer-test-key"
    assert api_base == "https://api.kaer.ai/v1"


def test_kaer_provider_reads_api_base_override(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("KAER_API_KEY", "kaer-test-key")
    monkeypatch.setenv("KAER_API_BASE", "https://kaer.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="kaer/janus-1-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "kaer"
    assert api_base == "https://kaer.internal.example/v1"


def test_kaer_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("KAER_API_KEY", "kaer-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="kaer/janus-1-flash",
        custom_llm_provider=None,
        api_base="https://kaer.internal.example/v1",
        api_key="kaer-explicit-key",
    )

    assert provider == "kaer"
    assert api_key == "kaer-explicit-key"
    assert api_base == "https://kaer.internal.example/v1"


def test_kaer_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    kaer = next(provider for provider in providers if provider["litellm_provider"] == "kaer")

    assert kaer["provider_display_name"] == "Kaer"
    assert kaer["default_model_placeholder"] == "kaer/janus-1-flash"
    assert {field["key"]: field["required"] for field in kaer["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_kaer_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.kaer.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_kaer",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "janus-1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Kaer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="kaer/janus-1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=16,
            api_key="kaer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.kaer.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer kaer-test-key"
    assert body["model"] == "janus-1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 16
    assert response.choices[0].message.content == "Hello from Kaer"
    assert response.usage.total_tokens == 7


def test_kaer_streaming_chat_completion_request():
    chunks: Final = [
        {
            "id": "chatcmpl_kaer",
            "object": "chat.completion.chunk",
            "created": 1_789_550_000,
            "model": "janus-1-pro",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}],
        }
        for piece in ("Hello ", "from Kaer")
    ] + [
        {
            "id": "chatcmpl_kaer",
            "object": "chat.completion.chunk",
            "created": 1_789_550_000,
            "model": "janus-1-pro",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }
    ]
    sse: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.kaer.ai/v1/chat/completions").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode(),
        )
        stream: Final = litellm.completion(
            model="kaer/janus-1-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            stream_options={"include_usage": True},
            api_key="kaer-test-key",
        )
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "janus-1-pro"
    assert body["stream"] is True
    assert text == "Hello from Kaer"
