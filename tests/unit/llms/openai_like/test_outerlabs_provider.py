import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.outerlabs.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_outerlabs",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-sonnet-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Outer Labs"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": False,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
OUTERLABS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("outerlabs/")))


def test_outerlabs_is_a_registered_provider():
    assert litellm.LlmProviders.OUTERLABS.value == "outerlabs"
    assert "outerlabs" in litellm.provider_list
    assert "outerlabs" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_outerlabs_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OUTERLABS_API_KEY", "outerlabs-test-key")
    monkeypatch.delenv("OUTERLABS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="outerlabs/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "outerlabs"
    assert api_key == "outerlabs-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_outerlabs_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OUTERLABS_API_KEY", "outerlabs-env-key")
    monkeypatch.setenv("OUTERLABS_API_BASE", "https://outerlabs.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="outerlabs/claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://outerlabs.internal.example/v1",
        api_key="outerlabs-explicit-key",
    )

    assert provider == "outerlabs"
    assert api_key == "outerlabs-explicit-key"
    assert api_base == "https://outerlabs.internal.example/v1"


def test_outerlabs_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OUTERLABS_API_KEY", "outerlabs-env-key")
    monkeypatch.setenv("OUTERLABS_API_BASE", "https://outerlabs.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="outerlabs/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "outerlabs"
    assert api_base == "https://outerlabs.env.example/v1"


def test_outerlabs_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OUTERLABS_API_KEY", "outerlabs-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="claude-sonnet-5",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "outerlabs"
    assert api_key == "outerlabs-env-key"
    assert api_base == _DEFAULT_API_BASE


_BELOW_EVERY_PRICE_BAND: Final = 1_000
_PRICE_BANDS: Final = {
    "outerlabs/claude-haiku-5-5": "above_100k_tokens",
    "outerlabs/gpt-5.6-luna": "above_272k_tokens",
    "outerlabs/gpt-5.6-sol": "above_272k_tokens",
    "outerlabs/gpt-5.6-terra": "above_272k_tokens",
    "outerlabs/gpt-6-astra": "above_272k_tokens",
    "outerlabs/gpt-6-luna": "above_272k_tokens",
    "outerlabs/gpt-6-sol": "above_272k_tokens",
    "outerlabs/gpt-6.1-sol": "above_272k_tokens",
}
# Forced tool choice is refused upstream for these models
_NO_FORCED_TOOL_CHOICE: Final = frozenset(
    {
        "outerlabs/claude-fable-5-1",
        "outerlabs/claude-opus-5-5",
        "outerlabs/claude-sonnet-5-5",
        "outerlabs/gpt-6-luna",
    }
)


def test_outerlabs_lists_every_model():
    assert len(OUTERLABS_MODELS) == 16
    assert set(_PRICE_BANDS) <= set(OUTERLABS_MODELS)
    assert _NO_FORCED_TOOL_CHOICE <= set(OUTERLABS_MODELS)


@pytest.mark.parametrize("model", OUTERLABS_MODELS)
def test_outerlabs_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_computer_use,
        supports_function_calling,
        supports_pdf_input,
        supports_prompt_caching,
        supports_reasoning,
        supports_response_schema,
        supports_tool_choice,
        supports_vision,
        supports_web_search,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=_BELOW_EVERY_PRICE_BAND,
        completion_tokens=_BELOW_EVERY_PRICE_BAND,
        custom_llm_provider="outerlabs",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=_BELOW_EVERY_PRICE_BAND,
        completion_tokens=0,
        cache_read_input_tokens=_BELOW_EVERY_PRICE_BAND,
        custom_llm_provider="outerlabs",
    )
    cache_write_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=_BELOW_EVERY_PRICE_BAND,
        completion_tokens=0,
        cache_creation_input_tokens=_BELOW_EVERY_PRICE_BAND,
        custom_llm_provider="outerlabs",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * _BELOW_EVERY_PRICE_BAND)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * _BELOW_EVERY_PRICE_BAND)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * _BELOW_EVERY_PRICE_BAND)
    assert cache_write_cost == pytest.approx(entry["cache_creation_input_token_cost"] * _BELOW_EVERY_PRICE_BAND)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "outerlabs"
    assert supports_function_calling(model) is True
    assert supports_reasoning(model) is True
    assert supports_prompt_caching(model) is True
    assert supports_response_schema(model) is True
    assert supports_tool_choice(model) is (model not in _NO_FORCED_TOOL_CHOICE)
    assert entry["supported_modalities"] == ["text"]
    # Not confirmed upstream, so the row must say so instead of inheriting the bare model row's flags
    assert supports_vision(model) is False
    assert supports_pdf_input(model) is False
    assert supports_web_search(model) is False
    assert supports_computer_use(model) is False


@pytest.mark.parametrize("model", sorted(_PRICE_BANDS))
def test_outerlabs_long_prompts_use_the_higher_price_band(model: str):
    from litellm.cost_calculator import cost_per_token

    entry: Final = litellm.model_cost[model]
    band: Final = _PRICE_BANDS[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=500_000,
        completion_tokens=1_000,
        custom_llm_provider="outerlabs",
    )

    assert entry[f"input_cost_per_token_{band}"] > entry["input_cost_per_token"]
    assert prompt_cost == pytest.approx(entry[f"input_cost_per_token_{band}"] * 500_000)
    assert completion_cost == pytest.approx(entry[f"output_cost_per_token_{band}"] * 1_000)


@pytest.mark.usefixtures("local_model_cost_map")
def test_outerlabs_wildcard_expands_to_every_outerlabs_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("outerlabs/*")) == set(OUTERLABS_MODELS)


def test_outerlabs_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("outerlabs/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("outerlabs/a-model-not-in-the-cost-map") is True


def test_outerlabs_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    outerlabs_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("outerlabs/")}

    assert outerlabs_entries
    assert tuple(sorted(outerlabs_entries)) == OUTERLABS_MODELS
    assert outerlabs_entries == {name: backup[name] for name in outerlabs_entries}


def test_outerlabs_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    outerlabs: Final = next(provider for provider in providers if provider["litellm_provider"] == "outerlabs")

    assert outerlabs["provider"] == "OUTERLABS"
    assert outerlabs["provider_display_name"] == "Outer Labs"
    assert outerlabs["default_model_placeholder"] in OUTERLABS_MODELS
    assert {field["key"]: field["required"] for field in outerlabs["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_outerlabs_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["outerlabs"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["outerlabs"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["low", "max"])
def test_outerlabs_chat_completion_request(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="outerlabs/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="outerlabs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer outerlabs-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from Outer Labs"


def test_outerlabs_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="outerlabs/claude-sonnet-5",
            input="Say hello",
            api_key="outerlabs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer outerlabs-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Outer Labs"


@pytest.mark.asyncio
async def test_outerlabs_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="outerlabs/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="outerlabs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer outerlabs-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Outer Labs" for block in response["content"])


def test_outerlabs_cache_control_marks_reach_upstream(monkeypatch: pytest.MonkeyPatch):
    system_block: Final = {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="outerlabs/claude-sonnet-5",
            messages=[{"role": "system", "content": [system_block]}, {"role": "user", "content": "Say hello"}],
            api_key="outerlabs-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["messages"][0]["content"] == [system_block]


def test_outerlabs_missing_key_never_falls_back_to_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OUTERLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-secret")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="OUTERLABS_API_KEY"):
            litellm.completion(model="outerlabs/claude-sonnet-5", messages=[{"role": "user", "content": "Say hello"}])
    assert route.call_count == 0
