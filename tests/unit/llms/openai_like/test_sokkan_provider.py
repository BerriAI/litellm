import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

# Rates published by https://api.sokkan.ch/v1/models, stated independently of the cost map.
EXPECTED_SOKKAN_MODELS: Final = {
    "sokkan/openai/gpt-oss-20b": {
        "input_cost_per_token": 4e-08,
        "output_cost_per_token": 1.5e-07,
        "max_input_tokens": 65536,
        "max_output_tokens": 32768,
        "supports_reasoning": True,
    },
    "sokkan/qwen/qwen3-coder-30b-a3b-instruct": {
        "input_cost_per_token": 7e-08,
        "output_cost_per_token": 2.7e-07,
        "max_input_tokens": 32768,
        "max_output_tokens": 16384,
        "supports_reasoning": False,
    },
}


def test_sokkan_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SOKKAN_API_KEY", "sokkan-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="sokkan/qwen/qwen3-coder-30b-a3b-instruct",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen/qwen3-coder-30b-a3b-instruct"
    assert provider == "sokkan"
    assert api_key == "sokkan-test-key"
    assert api_base == "https://api.sokkan.ch/v1"


def test_sokkan_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SOKKAN_API_KEY", "sokkan-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="sokkan/openai/gpt-oss-20b",
        custom_llm_provider=None,
        api_base="https://sokkan.internal.example/v1",
        api_key="sokkan-explicit-key",
    )

    assert provider == "sokkan"
    assert api_key == "sokkan-explicit-key"
    assert api_base == "https://sokkan.internal.example/v1"


def test_sokkan_cost_map_lists_exactly_the_expected_models():
    assert sorted(name for name in litellm.model_cost if name.startswith("sokkan/")) == sorted(EXPECTED_SOKKAN_MODELS)


@pytest.mark.parametrize("model,expected", sorted(EXPECTED_SOKKAN_MODELS.items()))
def test_sokkan_model_cost_and_capabilities(model: str, expected: dict):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="sokkan",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(expected["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(expected["output_cost_per_token"] * 1_000_000)
    assert model_info["max_input_tokens"] == expected["max_input_tokens"]
    assert model_info["max_tokens"] == model_info["max_output_tokens"] == expected["max_output_tokens"]
    assert model_info["supports_reasoning"] is expected["supports_reasoning"]
    assert model_info["litellm_provider"] == "sokkan"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_native_streaming"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is False


def test_sokkan_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    sokkan_entries = {name: entry for name, entry in cost_map.items() if name.startswith("sokkan/")}

    assert sorted(sokkan_entries) == sorted(EXPECTED_SOKKAN_MODELS)
    assert sokkan_entries == {name: backup[name] for name in sokkan_entries}


def test_sokkan_chat_completion_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sokkan.ch/v1/chat/completions").respond(
            200,
            json={
                "id": "sk-test",
                "object": "chat.completion",
                "created": 1_790_000_000,
                "model": "qwen/qwen3-coder-30b-a3b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from SOKKAN"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="sokkan/qwen/qwen3-coder-30b-a3b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sokkan-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.sokkan.ch/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sokkan-test-key"
    assert body["model"] == "qwen/qwen3-coder-30b-a3b-instruct"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from SOKKAN"
