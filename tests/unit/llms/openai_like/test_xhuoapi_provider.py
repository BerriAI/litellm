import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.xhuoapi.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_xhuoapi",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4.1-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from XHuoAPI", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "text_completion": False,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
XHUOAPI_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("xhuoapi/")))
FLAT_PRICED_MODELS: Final = tuple(name for name in XHUOAPI_MODELS if "tiered_pricing" not in litellm.model_cost[name])
TIERED_MODELS: Final = tuple(name for name in XHUOAPI_MODELS if "tiered_pricing" in litellm.model_cost[name])
OFF_PEAK_MODELS: Final = tuple(name for name in XHUOAPI_MODELS if "off_peak_pricing" in litellm.model_cost[name])


def test_xhuoapi_is_a_registered_provider():
    assert litellm.LlmProviders.XHUOAPI.value == "xhuoapi"
    assert "xhuoapi" in litellm.provider_list
    assert "xhuoapi" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_xhuoapi_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XHUOAPI_API_KEY", "xhuoapi-test-key")
    monkeypatch.delenv("XHUOAPI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="xhuoapi/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "xhuoapi"
    assert api_key == "xhuoapi-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_xhuoapi_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XHUOAPI_API_KEY", "xhuoapi-env-key")
    monkeypatch.setenv("XHUOAPI_API_BASE", "https://xhuoapi.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="xhuoapi/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://xhuoapi.internal.example/v1",
        api_key="xhuoapi-explicit-key",
    )

    assert provider == "xhuoapi"
    assert api_key == "xhuoapi-explicit-key"
    assert api_base == "https://xhuoapi.internal.example/v1"


def test_xhuoapi_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XHUOAPI_API_KEY", "xhuoapi-env-key")
    monkeypatch.setenv("XHUOAPI_API_BASE", "https://xhuoapi.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="xhuoapi/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "xhuoapi"
    assert api_base == "https://xhuoapi.env.example/v1"


def test_xhuoapi_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XHUOAPI_API_KEY", "xhuoapi-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "xhuoapi"
    assert api_key == "xhuoapi-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", FLAT_PRICED_MODELS)
def test_xhuoapi_model_cost_and_capabilities(model: str):
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
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="xhuoapi",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="xhuoapi",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "xhuoapi"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_prompt_caching(model) is entry["supports_prompt_caching"]
    assert supports_response_schema(model) is entry["supports_response_schema"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.parametrize("model", TIERED_MODELS)
def test_xhuoapi_tiered_model_bills_each_request_at_its_tier(model: str):
    from litellm.cost_calculator import cost_per_token

    low_tier, high_tier = litellm.model_cost[model]["tiered_pricing"]
    small_prompt: Final = int(low_tier["range"][1]) // 2
    large_prompt: Final = int(high_tier["range"][0]) + 1_000

    small_cost, _ = cost_per_token(
        model=model, prompt_tokens=small_prompt, completion_tokens=0, custom_llm_provider="xhuoapi"
    )
    large_cost, _ = cost_per_token(
        model=model, prompt_tokens=large_prompt, completion_tokens=0, custom_llm_provider="xhuoapi"
    )

    assert small_cost == pytest.approx(small_prompt * low_tier["input_cost_per_token"])
    assert large_cost == pytest.approx(large_prompt * high_tier["input_cost_per_token"])


@pytest.mark.parametrize("model", OFF_PEAK_MODELS)
def test_xhuoapi_deepseek_off_peak_windows_match_deepseek(model: str):
    deepseek_windows: Final = litellm.model_cost["deepseek/deepseek-flash"]["off_peak_pricing"]["windows"]
    off_peak: Final = litellm.model_cost[model]["off_peak_pricing"]

    assert off_peak["windows"] == deepseek_windows
    assert off_peak["input_cost_per_token"] < litellm.model_cost[model]["input_cost_per_token"]
    assert off_peak["output_cost_per_token"] < litellm.model_cost[model]["output_cost_per_token"]


@pytest.mark.usefixtures("local_model_cost_map")
def test_xhuoapi_wildcard_expands_to_every_xhuoapi_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("xhuoapi/*")) == set(XHUOAPI_MODELS)


def test_xhuoapi_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("xhuoapi/a-model-not-in-the-cost-map") is True


def test_xhuoapi_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    xhuoapi_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("xhuoapi/")}

    assert xhuoapi_entries
    assert tuple(sorted(xhuoapi_entries)) == XHUOAPI_MODELS
    assert xhuoapi_entries == {name: backup[name] for name in xhuoapi_entries}


def test_xhuoapi_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    xhuoapi: Final = next(provider for provider in providers if provider["litellm_provider"] == "xhuoapi")

    assert xhuoapi["provider"] == "XHUOAPI"
    assert xhuoapi["provider_display_name"] == "XHuoAPI"
    assert xhuoapi["default_model_placeholder"] in XHUOAPI_MODELS
    assert {field["key"]: field["required"] for field in xhuoapi["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_xhuoapi_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["xhuoapi"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["xhuoapi"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_xhuoapi_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="xhuoapi/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="xhuoapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer xhuoapi-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from XHuoAPI"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_xhuoapi_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="xhuoapi/deepseek-v4.1-flash",
            input="Say hello",
            api_key="xhuoapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer xhuoapi-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from XHuoAPI"


@pytest.mark.asyncio
async def test_xhuoapi_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="xhuoapi/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="xhuoapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer xhuoapi-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from XHuoAPI" for block in response["content"])
