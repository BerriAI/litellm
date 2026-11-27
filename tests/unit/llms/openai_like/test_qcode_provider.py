import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.qcode.cc/openai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_qcode",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-6.2-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from QCode", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
QCODE_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("qcode/")))


def test_qcode_is_a_registered_provider():
    assert litellm.LlmProviders.QCODE.value == "qcode"
    assert "qcode" in litellm.provider_list
    assert "qcode" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_qcode_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QCODE_API_KEY", "qcode-test-key")
    monkeypatch.delenv("QCODE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="qcode/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-6.2-sol"
    assert provider == "qcode"
    assert api_key == "qcode-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_qcode_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QCODE_API_KEY", "qcode-env-key")
    monkeypatch.setenv("QCODE_API_BASE", "https://qcode.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="qcode/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base="https://qcode.internal.example/v1",
        api_key="qcode-explicit-key",
    )

    assert provider == "qcode"
    assert api_key == "qcode-explicit-key"
    assert api_base == "https://qcode.internal.example/v1"


def test_qcode_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QCODE_API_KEY", "qcode-env-key")
    monkeypatch.setenv("QCODE_API_BASE", "https://qcode.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="qcode/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "qcode"
    assert api_base == "https://qcode.env.example/v1"


def test_qcode_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QCODE_API_KEY", "qcode-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-6.2-sol"
    assert provider == "qcode"
    assert api_key == "qcode-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", QCODE_MODELS)
def test_qcode_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_response_schema,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=100_000,
        completion_tokens=100_000,
        custom_llm_provider="qcode",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=100_000,
        completion_tokens=0,
        cache_read_input_tokens=100_000,
        custom_llm_provider="qcode",
    )
    long_prompt_cost, long_completion_cost = cost_per_token(
        model=model,
        prompt_tokens=300_000,
        completion_tokens=10_000,
        custom_llm_provider="qcode",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 100_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 100_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 100_000)
    assert long_prompt_cost == pytest.approx(entry["input_cost_per_token_above_272k_tokens"] * 300_000)
    assert long_completion_cost == pytest.approx(entry["output_cost_per_token_above_272k_tokens"] * 10_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "qcode"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_prompt_caching(model) is entry["supports_prompt_caching"]
    assert supports_response_schema(model) is entry["supports_response_schema"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_qcode_wildcard_expands_to_every_qcode_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("qcode/*")) == set(QCODE_MODELS)


def test_qcode_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("qcode/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("qcode/a-model-not-in-the-cost-map") is True


def test_qcode_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    qcode_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("qcode/")}

    assert qcode_entries
    assert tuple(sorted(qcode_entries)) == QCODE_MODELS
    assert qcode_entries == {name: backup[name] for name in qcode_entries}


def test_qcode_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    qcode: Final = next(provider for provider in providers if provider["litellm_provider"] == "qcode")

    assert qcode["provider"] == "QCODE"
    assert qcode["provider_display_name"] == "QCode"
    assert qcode["default_model_placeholder"] in QCODE_MODELS
    assert {field["key"]: field["required"] for field in qcode["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_qcode_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["qcode"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["qcode"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["none", "max"])
def test_qcode_chat_completion_request(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="qcode/gpt-6.2-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="qcode-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer qcode-test-key"
    assert body["model"] == "gpt-6.2-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from QCode"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_qcode_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="qcode/gpt-6.2-sol",
            input="Say hello",
            api_key="qcode-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer qcode-test-key"
    assert body["model"] == "gpt-6.2-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from QCode"


@pytest.mark.asyncio
async def test_qcode_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="qcode/gpt-6.2-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="qcode-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer qcode-test-key"
    assert body["model"] == "gpt-6.2-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from QCode" for block in response["content"])
