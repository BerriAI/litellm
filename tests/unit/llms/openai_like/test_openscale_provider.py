import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_openscale_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENSCALE_API_KEY", "openscale-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="openscale/deepseek-v4-1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-1-flash"
    assert provider == "openscale"
    assert api_key == "openscale-test-key"
    assert api_base == "https://api.openscale.so/v1"


def test_openscale_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENSCALE_API_KEY", "openscale-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="openscale/glm-5-3-flash",
        custom_llm_provider=None,
        api_base="https://openscale.internal.example/v1",
        api_key="openscale-explicit-key",
    )

    assert provider == "openscale"
    assert api_key == "openscale-explicit-key"
    assert api_base == "https://openscale.internal.example/v1"


def test_openscale_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    openscale = next(provider for provider in providers if provider["litellm_provider"] == "openscale")

    assert openscale["provider"] == "OPENSCALE"
    assert openscale["provider_display_name"] == "OpenScale"
    assert openscale["default_model_placeholder"] == "openscale/deepseek-v4-1-flash"
    assert {field["key"]: field["required"] for field in openscale["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_openscale_supports_tools_and_reasoning_effort():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config: Final = create_config_class(JSONProviderRegistry.get("openscale"))()
    supported: Final = config.get_supported_openai_params("deepseek-v4-1-flash")

    assert "tools" in supported
    assert "tool_choice" in supported
    assert "reasoning_effort" in supported


def test_openscale_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.openscale.so/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_openscale",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "deepseek-v4-1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenScale"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="openscale/deepseek-v4-1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="openscale-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.openscale.so/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer openscale-test-key"
    assert body["model"] == "deepseek-v4-1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from OpenScale"
