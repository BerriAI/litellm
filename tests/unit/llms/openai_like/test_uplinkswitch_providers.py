import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import LlmProviders, ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASES: Final = {
    "uplinkswitch": "https://api.uplinkswitch.com/v1",
    "relayswitch": "https://api.relayswitch.io/v1",
    "uncensoredllm": "https://api.uncensoredllm.ai/v1",
    "pentestllm": "https://api.pentestllm.ai/v1",
}
_PROVIDERS: Final = tuple(_DEFAULT_API_BASES)
_MODELS: Final = {
    provider: tuple(sorted(name for name in litellm.model_cost if name.startswith(f"{provider}/")))
    for provider in _PROVIDERS
}
_PRICED_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.split("/", 1)[0] in _PROVIDERS))
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_gateway",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "wire-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
}
_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _package_json(relative_path: str) -> object:
    return json.loads((Path(litellm.__file__).parent / relative_path).read_text())


def _repo_json(relative_path: str) -> object:
    return json.loads((Path(litellm.__file__).parent.parent / relative_path).read_text())


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_provider_is_registered_with_its_priced_models(provider: str):
    assert LlmProviders(provider).value == provider
    assert provider in litellm.provider_list
    assert provider in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASES[provider] in litellm.constants.openai_compatible_endpoints
    assert _MODELS[provider]
    assert set(litellm.models_by_provider[provider]) == set(_MODELS[provider])


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_provider_resolves_to_its_own_key_and_default_base(provider: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", f"{provider}-env-key")
    monkeypatch.delenv(f"{provider.upper()}_API_BASE", raising=False)

    model, resolved_provider, api_key, api_base = get_llm_provider(
        model=f"{provider}/wire-model", custom_llm_provider=None, api_base=None, api_key=None
    )

    assert (model, resolved_provider) == ("wire-model", provider)
    assert api_key == f"{provider}-env-key"
    assert api_base == _DEFAULT_API_BASES[provider]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_api_base_env_overrides_the_default_base(provider: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", f"{provider}-env-key")
    monkeypatch.setenv(f"{provider.upper()}_API_BASE", "https://gateway.env.example/v1")

    _, resolved_provider, _, api_base = get_llm_provider(
        model=f"{provider}/wire-model", custom_llm_provider=None, api_base=None, api_key=None
    )

    assert resolved_provider == provider
    assert api_base == "https://gateway.env.example/v1"


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_default_base_autodetects_the_provider_and_its_key(provider: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", f"{provider}-env-key")

    model, resolved_provider, api_key, _ = get_llm_provider(
        model="wire-model", custom_llm_provider=None, api_base=_DEFAULT_API_BASES[provider], api_key=None
    )

    assert (model, resolved_provider) == ("wire-model", provider)
    assert api_key == f"{provider}-env-key"


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_chat_request_goes_to_the_provider_base_with_openai_params_unchanged(provider: str):
    url: Final = f"{_DEFAULT_API_BASES[provider]}/chat/completions"
    with respx.mock() as upstream:
        route: Final = upstream.post(url).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"{provider}/wire-model",
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=64,
            tools=[_TOOL],
            api_key=f"{provider}-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == f"Bearer {provider}-test-key"
    assert body["model"] == "wire-model"
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    assert body["tools"] == [_TOOL]
    assert response.choices[0].message.content == "pong"


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_responses_request_is_bridged_to_chat_completions(provider: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASES[provider]}/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.responses(
            model=f"{provider}/wire-model", input="ping", api_key=f"{provider}-test-key"
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert [item.content[0].text for item in response.output if item.type == "message"] == ["pong"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", _PROVIDERS)
async def test_anthropic_messages_request_is_bridged_to_chat_completions(
    provider: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASES[provider]}/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"{provider}/wire-model",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=32,
            api_key=f"{provider}-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "pong" for block in response["content"])


@pytest.mark.parametrize("model", _PRICED_MODELS)
def test_priced_model_bills_its_cost_map_rates(model: str):
    rates: Final = litellm.model_cost[model]
    cost: Final = litellm.completion_cost(
        completion_response=ModelResponse(
            model=model,
            usage=Usage(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400_000),
            ),
        ),
        model=model,
        custom_llm_provider=model.split("/", 1)[0],
    )

    assert rates["cache_read_input_token_cost"] < rates["input_cost_per_token"]
    assert cost == pytest.approx(
        600_000 * rates["input_cost_per_token"]
        + 400_000 * rates["cache_read_input_token_cost"]
        + 1_000_000 * rates["output_cost_per_token"]
    )


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_wildcard_expands_to_every_priced_model(provider: str):
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard(f"{provider}/*")) == set(_MODELS[provider])


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_backup_cost_map_mirrors_the_cost_map(provider: str):
    cost_map: Final = _repo_json("model_prices_and_context_window.json")
    backup: Final = _package_json("model_prices_and_context_window_backup.json")
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith(f"{provider}/")}

    assert tuple(sorted(entries)) == _MODELS[provider]
    assert entries == {name: backup[name] for name in entries}


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_add_model_form_lists_the_provider_with_a_priced_default_model(provider: str):
    form: Final = next(
        entry
        for entry in _package_json("proxy/public_endpoints/provider_create_fields.json")
        if entry["litellm_provider"] == provider
    )

    assert form["provider"] == provider.upper()
    assert form["default_model_placeholder"] in _MODELS[provider]
    assert {field["key"]: field["required"] for field in form["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
    assert {field["key"]: field["placeholder"] for field in form["credential_fields"]} == {
        "api_base": _DEFAULT_API_BASES[provider],
        "api_key": None,
    }


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_endpoint_support_lists_chat_and_its_bridged_endpoints(provider: str):
    root: Final = _repo_json("provider_endpoints_support.json")["providers"][provider]["endpoints"]
    backup: Final = _package_json("provider_endpoints_support_backup.json")["providers"][provider]["endpoints"]

    assert backup == root
    assert (root["chat_completions"], root["messages"], root["responses"], root["embeddings"]) == (
        True,
        True,
        True,
        False,
    )
