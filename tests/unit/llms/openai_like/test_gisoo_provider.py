import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm

_CHAT_URL: Final = "https://gisoo.pro/api/v1/chat/completions"
_CHAT_RESPONSE: Final = {
    "id": "chatcmpl_gisoo",
    "object": "chat.completion",
    "created": 1_792_000_000,
    "model": "claude-sonnet-5-5",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from Gisoo"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_gisoo_is_a_registered_provider():
    assert litellm.LlmProviders.GISOO.value == "gisoo"
    assert "gisoo" in litellm.provider_list
    assert "gisoo" in litellm.constants.openai_compatible_providers


def test_gisoo_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.delenv("GISOO_API_BASE", raising=False)
    monkeypatch.setenv("GISOO_API_KEY", "gisoo-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gisoo/claude-sonnet-5-5", custom_llm_provider=None, api_base=None, api_key=None
    )

    assert model == "claude-sonnet-5-5"
    assert provider == "gisoo"
    assert api_key == "gisoo-test-key"
    assert api_base == "https://gisoo.pro/api/v1"


def test_gisoo_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("GISOO_API_KEY", "gisoo-test-key")
    monkeypatch.setenv("GISOO_API_BASE", "https://gisoo.example/api/v1")

    _, _, _, api_base = get_llm_provider(
        model="gisoo/claude-sonnet-5-5", custom_llm_provider=None, api_base=None, api_key=None
    )

    assert api_base == "https://gisoo.example/api/v1"


def test_gisoo_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_RESPONSE)
        response: Final = litellm.completion(
            model="gisoo/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="gisoo-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert str(request.url) == _CHAT_URL
    assert request.headers["authorization"] == "Bearer gisoo-test-key"
    assert body["model"] == "claude-sonnet-5-5"
    assert response.choices[0].message.content == "Hello from Gisoo"


def test_gisoo_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://gisoo.pro/api/v1/responses").respond(
            200,
            json={
                "id": "resp_gisoo",
                "object": "response",
                "created_at": 1_792_000_000,
                "model": "claude-sonnet-5-5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_gisoo",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Gisoo", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(model="gisoo/claude-sonnet-5-5", input="Say hello", api_key="gisoo-test-key")

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "claude-sonnet-5-5"
    assert response.output[0].content[0].text == "Hello from Gisoo"


def test_gisoo_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    gisoo = next(p for p in json.loads(fields_path.read_text()) if p["litellm_provider"] == "gisoo")

    assert gisoo["provider"] == "GISOO"
    assert {f["key"]: f["required"] for f in gisoo["credential_fields"]} == {"api_base": False, "api_key": True}


def test_gisoo_supported_endpoints():
    root = Path(litellm.__file__).parent
    for path in (root.parent / "provider_endpoints_support.json", root / "provider_endpoints_support_backup.json"):
        endpoints = json.loads(path.read_text())["providers"]["gisoo"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is True
        assert endpoints["messages"] is False
