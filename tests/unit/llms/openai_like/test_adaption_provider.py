import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.prod.adaptionlabs.ai/api/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_RESPONSES_URL: Final = f"{_DEFAULT_API_BASE}/responses"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_adaption",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "glm-5.2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Adaption"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "responses": True,
    "embeddings": False,
    "image_edits": False,
    "realtime": False,
    "video_generations": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
_PRICES_PER_MILLION: Final = {
    "adaption/glm-5.3-flash": (0.15, 0.03, 0.50),
    "adaption/glm-5.2": (1.40, 0.26, 4.40),
    "adaption/qwen3.5-397b": (0.55, 0.225, 3.50),
    "adaption/gemma-4-26b-a4b-it": (0.0765, 0.0425, 0.255),
    "adaption/qwen3.8-27b": (0.039, 0.0039, 1.25),
    "adaption/deepseek-v4-flash": (0.04444, 0.001414, 0.13332),
    "adaption/gemma-4-31b-it": (0.09, 0.05, 0.34),
}
_MAX_OUTPUT_TOKENS: Final = {
    "adaption/glm-5.3-flash": 131_072,
    "adaption/glm-5.2": 131_072,
    "adaption/qwen3.5-397b": 65_536,
    "adaption/gemma-4-26b-a4b-it": 32_768,
    "adaption/qwen3.8-27b": 32_768,
    "adaption/deepseek-v4-flash": 384_000,
    "adaption/gemma-4-31b-it": 32_768,
}
_VISION_MODELS: Final = frozenset(
    {
        "adaption/glm-5.3-flash",
        "adaption/qwen3.5-397b",
        "adaption/gemma-4-26b-a4b-it",
        "adaption/qwen3.8-27b",
        "adaption/gemma-4-31b-it",
    }
)
ADAPTION_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("adaption/")))


def test_adaption_is_a_registered_provider():
    assert litellm.LlmProviders.ADAPTION.value == "adaption"
    assert "adaption" in litellm.provider_list
    assert "adaption" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_adaption_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADAPTION_API_KEY", "adaption-test-key")
    monkeypatch.delenv("ADAPTION_API_BASE", raising=False)

    assert get_llm_provider(model="adaption/glm-5.2") == (
        "glm-5.2",
        "adaption",
        "adaption-test-key",
        _DEFAULT_API_BASE,
    )


def test_adaption_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADAPTION_API_KEY", "adaption-env-key")
    monkeypatch.setenv("ADAPTION_API_BASE", "https://adaption.env.example/v1")

    assert get_llm_provider(
        model="adaption/glm-5.2", api_key="adaption-explicit-key", api_base="https://gateway.example/v1"
    ) == ("glm-5.2", "adaption", "adaption-explicit-key", "https://gateway.example/v1")


def test_adaption_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADAPTION_API_KEY", "adaption-env-key")
    monkeypatch.setenv("ADAPTION_API_BASE", "https://adaption.env.example/v1")

    _, provider, _, api_base = get_llm_provider(model="adaption/glm-5.2")

    assert provider == "adaption"
    assert api_base == "https://adaption.env.example/v1"


def test_adaption_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADAPTION_API_KEY", "adaption-env-key")

    assert get_llm_provider(model="glm-5.2", api_base=_DEFAULT_API_BASE) == (
        "glm-5.2",
        "adaption",
        "adaption-env-key",
        _DEFAULT_API_BASE,
    )


@pytest.mark.parametrize("model", ADAPTION_MODELS)
def test_adaption_model_info_and_capabilities(model: str):
    from litellm.utils import supports_function_calling, supports_reasoning, supports_tool_choice, supports_vision

    entry: Final = litellm.model_cost[model]
    model_info: Final = litellm.get_model_info(model)

    assert model_info["litellm_provider"] == "adaption"
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert supports_function_calling(model) is True
    assert supports_tool_choice(model) is True
    assert supports_reasoning(model) is True
    assert supports_vision(model) is (model in _VISION_MODELS)
    assert entry["max_output_tokens"] == _MAX_OUTPUT_TOKENS[model]


@pytest.mark.parametrize("model", ADAPTION_MODELS)
def test_adaption_model_prices(model: str):
    input_price, cache_read_price, output_price = _PRICES_PER_MILLION[model]
    entry: Final = litellm.model_cost[model]

    assert entry["input_cost_per_token"] == pytest.approx(input_price / 1e6)
    assert entry["cache_read_input_token_cost"] == pytest.approx(cache_read_price / 1e6)
    assert entry["output_cost_per_token"] == pytest.approx(output_price / 1e6)


def test_adaption_cost_map_lists_every_model():
    assert ADAPTION_MODELS == tuple(
        sorted(
            f"adaption/{name}"
            for name in (
                "glm-5.3-flash",
                "glm-5.2",
                "qwen3.5-397b",
                "gemma-4-26b-a4b-it",
                "qwen3.8-27b",
                "deepseek-v4-flash",
                "gemma-4-31b-it",
            )
        )
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_adaption_wildcard_expands_to_every_adaption_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("adaption/*")) == set(ADAPTION_MODELS)


def test_adaption_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("adaption/a-model-not-in-the-cost-map") is True


def test_adaption_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("adaption/")}

    assert tuple(sorted(entries)) == ADAPTION_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_adaption_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    adaption: Final = next(provider for provider in providers if provider["litellm_provider"] == "adaption")

    assert adaption["provider"] == "ADAPTION"
    assert adaption["provider_display_name"] == "Adaption"
    assert adaption["default_model_placeholder"] in ADAPTION_MODELS
    assert {field["key"]: field["required"] for field in adaption["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_adaption_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["adaption"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["adaption"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_adaption_chat_completion_request_forwards_tools():
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="adaption/glm-5.2",
            messages=[{"role": "user", "content": "Say hello"}],
            tools=tools,
            tool_choice="auto",
            api_key="adaption-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer adaption-test-key"
    assert body["model"] == "glm-5.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert response.choices[0].message.content == "Hello from Adaption"


def test_adaption_uses_native_responses_endpoint():
    with respx.mock() as upstream:
        route: Final = upstream.post(_RESPONSES_URL).respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "glm-5.2",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="adaption/glm-5.2",
            api_key="adaption-test-key",
            input="Say hello",
            store=False,
        )

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer adaption-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == "glm-5.2"
    assert response.output[0].content[0].text == "hello"
