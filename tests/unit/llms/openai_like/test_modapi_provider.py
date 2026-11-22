import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://modapi.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_modapi",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-4.1-nano",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from ModAPI"},
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
MODAPI_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("modapi/")))
_EXPECTED_MODELS: Final = frozenset(
    f"modapi/{name}"
    for name in (
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5.2",
        "gpt-5.4-mini",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
    )
)
# ModAPI LLM_Standard prices in USD per 1M tokens: input, output, cache read, 5-minute cache write, 1-hour cache write.
_EXPECTED_PRICES: Final = {
    "modapi/gpt-4.1-nano": (0.1, 0.4, 0.025, None, None),
    "modapi/gpt-5.2": (1.75, 14.0, 0.175, None, None),
    "modapi/claude-sonnet-5-5": (2.0, 10.0, 0.1, 2.5, 4.0),
    "modapi/claude-opus-5-5": (4.0, 20.0, 0.2, 5.0, 8.0),
    "modapi/gemini-3.8-flash": (0.75, 3.75, 0.075, None, None),
}


def test_modapi_is_a_registered_provider():
    assert litellm.LlmProviders.MODAPI.value == "modapi"
    assert "modapi" in litellm.provider_list
    assert "modapi" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_modapi_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODAPI_API_KEY", "modapi-test-key")
    monkeypatch.delenv("MODAPI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="modapi/gpt-4.1-nano",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-4.1-nano"
    assert provider == "modapi"
    assert api_key == "modapi-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_modapi_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODAPI_API_KEY", "modapi-env-key")
    monkeypatch.setenv("MODAPI_API_BASE", "https://modapi.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="modapi/gpt-4.1-nano",
        custom_llm_provider=None,
        api_base="https://modapi.internal.example/v1",
        api_key="modapi-explicit-key",
    )

    assert provider == "modapi"
    assert api_key == "modapi-explicit-key"
    assert api_base == "https://modapi.internal.example/v1"


def test_modapi_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODAPI_API_KEY", "modapi-env-key")
    monkeypatch.setenv("MODAPI_API_BASE", "https://modapi.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="modapi/gpt-4.1-nano",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "modapi"
    assert api_base == "https://modapi.env.example/v1"


def test_modapi_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODAPI_API_KEY", "modapi-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-4.1-nano",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-4.1-nano"
    assert provider == "modapi"
    assert api_key == "modapi-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_modapi_ships_exactly_the_reviewed_models():
    assert set(MODAPI_MODELS) == _EXPECTED_MODELS


@pytest.mark.parametrize("model", MODAPI_MODELS)
def test_modapi_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_prompt_caching, supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="modapi",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    if "cache_read_input_token_cost" in entry:
        cached_prompt_cost, _ = cost_per_token(
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_input_tokens=1_000_000,
            custom_llm_provider="modapi",
        )
        assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert entry["source"] == f"https://modapi.ai/pricing?search={model.removeprefix('modapi/')}"
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "modapi"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_prompt_caching(model) is entry["supports_prompt_caching"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])
    assert set(entry["supported_modalities"]) <= {"text", "image"}
    assert entry["supported_output_modalities"] == ["text"]


@pytest.mark.parametrize("model", sorted(_EXPECTED_PRICES))
def test_modapi_prices_match_modapi_configuration(model: str):
    from litellm.cost_calculator import cost_per_token

    input_price, output_price, cache_read, cache_write_5m, cache_write_1h = _EXPECTED_PRICES[model]
    entry: Final = litellm.model_cost[model]
    assert entry["input_cost_per_token"] * 1_000_000 == pytest.approx(input_price)
    assert entry["output_cost_per_token"] * 1_000_000 == pytest.approx(output_price)
    assert entry["cache_read_input_token_cost"] * 1_000_000 == pytest.approx(cache_read)
    assert entry.get("cache_creation_input_token_cost") == (
        None if cache_write_5m is None else pytest.approx(cache_write_5m / 1_000_000)
    )
    assert entry.get("cache_creation_input_token_cost_above_1hr") == (
        None if cache_write_1h is None else pytest.approx(cache_write_1h / 1_000_000)
    )

    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="modapi",
    )
    assert cached_prompt_cost == pytest.approx(cache_read)


@pytest.mark.parametrize(
    ("model", "write_5m", "write_1h"),
    [("modapi/claude-sonnet-5-5", 2.5, 4.0), ("modapi/claude-opus-5-5", 5.0, 8.0)],
)
def test_modapi_claude_cache_writes_are_billed_at_modapi_rates(model: str, write_5m: float, write_1h: float):
    from litellm.cost_calculator import cost_per_token
    from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage

    def prompt_cost_for(details: CacheCreationTokenDetails) -> float:
        usage = Usage(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cache_creation_tokens=1_000_000,
                cache_creation_token_details=details,
            ),
        )
        prompt_cost, _ = cost_per_token(model=model, custom_llm_provider="modapi", usage_object=usage)
        return prompt_cost

    assert prompt_cost_for(CacheCreationTokenDetails(ephemeral_5m_input_tokens=1_000_000)) == pytest.approx(write_5m)
    assert prompt_cost_for(CacheCreationTokenDetails(ephemeral_1h_input_tokens=1_000_000)) == pytest.approx(write_1h)


@pytest.mark.usefixtures("local_model_cost_map")
def test_modapi_wildcard_expands_to_every_modapi_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("modapi/*")) == set(MODAPI_MODELS)


def test_modapi_promises_no_capability_for_unlisted_models():
    from litellm.utils import supports_function_calling, supports_vision

    assert supports_function_calling("modapi/a-model-not-in-the-cost-map") is False
    assert supports_vision("modapi/a-model-not-in-the-cost-map") is False


def test_modapi_does_not_discover_models_over_http():
    """The model list is the reviewed static list; LiteLLM does not call ModAPI's GET /v1/models (which needs a key)."""
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.get(f"{_DEFAULT_API_BASE}/models").respond(200, json={"data": []})
        models: Final = litellm.get_valid_models(
            check_provider_endpoint=True, custom_llm_provider="modapi", api_key="modapi-test-key"
        )

    assert route.call_count == 0
    assert set(models) == _EXPECTED_MODELS


def test_modapi_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    modapi_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("modapi/")}

    assert modapi_entries
    assert tuple(sorted(modapi_entries)) == MODAPI_MODELS
    assert modapi_entries == {name: backup[name] for name in modapi_entries}


def test_modapi_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    modapi: Final = next(provider for provider in providers if provider["litellm_provider"] == "modapi")

    assert modapi["provider"] == "MODAPI"
    assert modapi["provider_display_name"] == "ModAPI"
    assert modapi["default_model_placeholder"] in MODAPI_MODELS
    assert {field["key"]: field["required"] for field in modapi["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_modapi_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["modapi"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["modapi"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_modapi_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="modapi/gpt-4.1-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="modapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer modapi-test-key"
    assert body["model"] == "gpt-4.1-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from ModAPI"


def test_modapi_text_completion_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.text_completion(
            model="modapi/gpt-4.1-nano",
            prompt="Say hello",
            api_key="modapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer modapi-test-key"
    assert body["model"] == "gpt-4.1-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].text == "Hello from ModAPI"


def test_modapi_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="modapi/gpt-4.1-nano",
            input="Say hello",
            api_key="modapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer modapi-test-key"
    assert body["model"] == "gpt-4.1-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from ModAPI"


@pytest.mark.asyncio
async def test_modapi_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="modapi/gpt-4.1-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="modapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer modapi-test-key"
    assert body["model"] == "gpt-4.1-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from ModAPI" for block in response["content"])
