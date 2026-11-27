import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.levanto.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_ROUTED_MODEL: Final = "deepseek/deepseek-v4-flash"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_levanto",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _ROUTED_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Levanto"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000},
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
LEVANTO_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("levanto/")))
LEVANTO_PRICED_MODELS: Final = tuple(
    name for name in LEVANTO_MODELS if "input_cost_per_token" in litellm.model_cost[name]
)


def _rates_for_one_million_in_and_out(model: str) -> float:
    entry: Final = litellm.model_cost[model]
    return 1_000_000 * (entry["input_cost_per_token"] + entry["output_cost_per_token"])


def test_levanto_is_a_registered_provider():
    assert litellm.LlmProviders.LEVANTO.value == "levanto"
    assert "levanto" in litellm.provider_list
    assert "levanto" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert LEVANTO_PRICED_MODELS
    assert set(litellm.models_by_provider["levanto"]) == set(LEVANTO_MODELS)


@pytest.mark.parametrize(
    ("model", "sent_model"),
    [("levanto/auto", "auto"), (f"levanto/{_ROUTED_MODEL}", _ROUTED_MODEL)],
)
def test_levanto_provider_resolution(monkeypatch: pytest.MonkeyPatch, model: str, sent_model: str):
    monkeypatch.setenv("LEVANTO_API_KEY", "lv_live_test")
    monkeypatch.delenv("LEVANTO_BASE_URL", raising=False)

    resolved_model, provider, api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert resolved_model == sent_model
    assert provider == "levanto"
    assert api_key == "lv_live_test"
    assert api_base == _DEFAULT_API_BASE


def test_levanto_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEVANTO_API_KEY", "lv_live_env")
    monkeypatch.setenv("LEVANTO_BASE_URL", "https://levanto.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="levanto/auto",
        custom_llm_provider=None,
        api_base="https://levanto.internal.example/v1",
        api_key="lv_live_explicit",
    )

    assert provider == "levanto"
    assert api_key == "lv_live_explicit"
    assert api_base == "https://levanto.internal.example/v1"


def test_levanto_base_url_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEVANTO_API_KEY", "lv_live_env")
    monkeypatch.setenv("LEVANTO_BASE_URL", "https://levanto.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="levanto/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "levanto"
    assert api_base == "https://levanto.env.example/v1"


def test_levanto_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEVANTO_API_KEY", "lv_live_env")

    model, provider, api_key, api_base = get_llm_provider(
        model="auto",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "auto"
    assert provider == "levanto"
    assert api_key == "lv_live_env"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", LEVANTO_PRICED_MODELS)
def test_levanto_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_prompt_caching

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="levanto",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="levanto",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["litellm_provider"] == "levanto"
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert supports_function_calling(model) is True
    assert supports_prompt_caching(model) is (entry["cache_read_input_token_cost"] < entry["input_cost_per_token"])


def test_levanto_auto_supports_tools_without_a_price_of_its_own():
    from litellm.utils import supports_function_calling, supports_tool_choice

    entry: Final = litellm.model_cost["levanto/auto"]

    assert "input_cost_per_token" not in entry
    assert supports_function_calling("levanto/auto") is True
    assert supports_tool_choice("levanto/auto") is True


def test_levanto_model_missing_from_cost_map_supports_tools():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("levanto/a-model-not-in-the-cost-map") is True


@pytest.mark.usefixtures("local_model_cost_map")
def test_levanto_wildcard_expands_to_every_levanto_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("levanto/*")) == set(LEVANTO_MODELS)


def test_levanto_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("levanto/")}

    assert tuple(sorted(entries)) == LEVANTO_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_levanto_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    levanto: Final = next(provider for provider in providers if provider["litellm_provider"] == "levanto")

    assert levanto["provider"] == "LEVANTO"
    assert levanto["default_model_placeholder"] in LEVANTO_MODELS
    assert {field["key"]: field["required"] for field in levanto["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_levanto_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["levanto"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["levanto"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_levanto_auto_request_sends_target_and_bills_the_model_that_answered():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="levanto/auto",
            messages=[{"role": "user", "content": "Say hello"}],
            target="economy",
            api_key="lv_live_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer lv_live_test"
    assert body["model"] == "auto"
    assert body["target"] == "economy"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.model == f"levanto/{_ROUTED_MODEL}"
    assert response.choices[0].message.content == "Hello from Levanto"
    assert response._hidden_params["response_cost"] == pytest.approx(
        _rates_for_one_million_in_and_out(f"levanto/{_ROUTED_MODEL}")
    )


@pytest.mark.parametrize("requested", ["auto", "served"])
@pytest.mark.parametrize("model", LEVANTO_PRICED_MODELS)
def test_levanto_call_bills_the_levanto_rate_of_the_served_model(model: str, requested: str):
    served: Final = model.removeprefix("levanto/")
    sent: Final = served if requested == "served" else requested
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json={**_CHAT_COMPLETION, "model": served})
        response: Final = litellm.completion(
            model=f"levanto/{sent}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="lv_live_test",
        )

    assert json.loads(route.calls.last.request.content)["model"] == sent
    assert response._hidden_params["response_cost"] == pytest.approx(_rates_for_one_million_in_and_out(model))


def test_levanto_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="levanto/auto",
            input="Say hello",
            api_key="lv_live_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer lv_live_test"
    assert body["model"] == "auto"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Levanto"


@pytest.mark.asyncio
async def test_levanto_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="levanto/auto",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="lv_live_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer lv_live_test"
    assert body["model"] == "auto"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Levanto" for block in response["content"])
