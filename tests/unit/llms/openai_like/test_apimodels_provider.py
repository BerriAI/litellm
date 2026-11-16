import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.apimodels.app/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
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
# A Tuesday inside the DeepSeek weekday peak hours, and two moments outside them.
_WEEKDAY_PEAK: Final = datetime(2026, 11, 3, 2, 0, tzinfo=timezone.utc)
_WEEKDAY_OFF_PEAK: Final = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
_WEEKEND: Final = datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)
APIMODELS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("apimodels/")))
OFF_PEAK_MODELS: Final = tuple(name for name in APIMODELS_MODELS if "off_peak_pricing" in litellm.model_cost[name])


def _chat_completion(model: str) -> dict:
    return {
        "id": "chatcmpl_apimodels",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from APIMODELS"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def test_apimodels_is_a_registered_provider():
    assert litellm.LlmProviders.APIMODELS.value == "apimodels"
    assert "apimodels" in litellm.provider_list
    assert "apimodels" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_apimodels_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-test-key")
    monkeypatch.delenv("APIMODELS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-5.6-luna"
    assert provider == "apimodels"
    assert api_key == "apimodels-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_apimodels_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.setenv("APIMODELS_API_BASE", "https://apimodels.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base="https://apimodels.internal.example/v1",
        api_key="apimodels-explicit-key",
    )

    assert provider == "apimodels"
    assert api_key == "apimodels-explicit-key"
    assert api_base == "https://apimodels.internal.example/v1"


def test_apimodels_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.setenv("APIMODELS_API_BASE", "https://apimodels.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "apimodels"
    assert api_base == "https://apimodels.env.example/v1"


def test_apimodels_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-5.6-luna",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-5.6-luna"
    assert provider == "apimodels"
    assert api_key == "apimodels-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", APIMODELS_MODELS)
def test_apimodels_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_prompt_caching, supports_reasoning, supports_vision

    entry: Final = litellm.model_cost[model]
    with pinned_billing_time(_WEEKDAY_PEAK):
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=1_000,
            completion_tokens=1_000,
            custom_llm_provider="apimodels",
        )
        cached_prompt_cost, _ = cost_per_token(
            model=model,
            prompt_tokens=1_000,
            completion_tokens=0,
            cache_read_input_tokens=1_000,
            custom_llm_provider="apimodels",
        )
    model_info: Final = litellm.get_model_info(model)

    assert entry["source"].startswith("https://apimodels.app/models/")
    assert entry["cache_read_input_token_cost"] < entry["input_cost_per_token"]
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000)
    assert model_info["litellm_provider"] == "apimodels"
    assert "max_tokens" not in entry
    assert model_info["max_tokens"] == entry.get("max_output_tokens")
    declared_capabilities: Final = {
        flag: check(model)
        for flag, check in (
            ("supports_function_calling", supports_function_calling),
            ("supports_reasoning", supports_reasoning),
            ("supports_prompt_caching", supports_prompt_caching),
            ("supports_vision", supports_vision),
        )
        if flag in entry
    }
    assert declared_capabilities == {flag: entry[flag] for flag in declared_capabilities}


@pytest.mark.usefixtures("local_model_cost_map")
def test_apimodels_wildcard_expands_to_every_apimodels_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("apimodels/*")) == set(APIMODELS_MODELS)


def test_apimodels_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    apimodels_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("apimodels/")}

    assert apimodels_entries
    assert tuple(sorted(apimodels_entries)) == APIMODELS_MODELS
    assert apimodels_entries == {name: backup[name] for name in apimodels_entries}


def test_apimodels_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    apimodels: Final = next(provider for provider in providers if provider["litellm_provider"] == "apimodels")

    assert apimodels["provider"] == "APIMODELS"
    assert apimodels["provider_display_name"] == "apimodels.app"
    assert apimodels["default_model_placeholder"] in APIMODELS_MODELS
    assert {field["key"]: field["required"] for field in apimodels["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_apimodels_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["apimodels"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["apimodels"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_apimodels_chat_completion_request_uses_env_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.delenv("APIMODELS_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_chat_completion("gpt-5.6-luna"))
        response: Final = litellm.completion(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer apimodels-env-key"
    assert body["model"] == "gpt-5.6-luna"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from APIMODELS"


def test_apimodels_chat_completion_request_honours_api_base_override():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://apimodels.internal.example/v1/chat/completions").respond(
            200, json=_chat_completion("gpt-5.6-luna")
        )
        response: Final = litellm.completion(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
            api_base="https://apimodels.internal.example/v1",
            api_key="apimodels-explicit-key",
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert str(request.url) == "https://apimodels.internal.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer apimodels-explicit-key"
    assert response.choices[0].message.content == "Hello from APIMODELS"


def test_apimodels_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_chat_completion("claude-opus-5-5"))
        response: Final = litellm.responses(
            model="apimodels/claude-opus-5-5",
            input="Say hello",
            api_key="apimodels-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer apimodels-test-key"
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from APIMODELS"


@pytest.mark.asyncio
async def test_apimodels_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_chat_completion("gpt-5.6-luna"))
        response: Final = await litellm.anthropic.messages.acreate(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="apimodels-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer apimodels-test-key"
    assert body["model"] == "gpt-5.6-luna"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from APIMODELS" for block in response["content"])


def _completion_with_usage(model: str, usage: dict[str, object]) -> litellm.ModelResponse:
    body: Final = {**_chat_completion(model), "usage": usage}
    with respx.mock() as upstream:
        upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=body)
        return litellm.completion(
            model=f"apimodels/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="apimodels-test-key",
        )


def test_apimodels_chat_completion_cost_bills_cached_tokens_at_cache_read_rate():
    row: Final = litellm.model_cost["apimodels/gpt-6-sol"]
    response: Final = _completion_with_usage(
        "gpt-6-sol",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
    )

    assert row["cache_read_input_token_cost"] < row["input_cost_per_token"]
    assert litellm.completion_cost(completion_response=response, model="apimodels/gpt-6-sol") == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )


def test_apimodels_prompt_over_272k_bills_the_whole_request_at_the_long_context_rates():
    row: Final = litellm.model_cost["apimodels/gpt-6-sol"]
    response: Final = _completion_with_usage(
        "gpt-6-sol",
        {
            "prompt_tokens": 300_000,
            "completion_tokens": 20,
            "total_tokens": 300_020,
            "prompt_tokens_details": {"cached_tokens": 100_000},
        },
    )

    assert row["input_cost_per_token_above_272k_tokens"] > row["input_cost_per_token"]
    assert litellm.completion_cost(completion_response=response, model="apimodels/gpt-6-sol") == pytest.approx(
        200_000 * row["input_cost_per_token_above_272k_tokens"]
        + 100_000 * row["cache_read_input_token_cost_above_272k_tokens"]
        + 20 * row["output_cost_per_token_above_272k_tokens"]
    )


@pytest.mark.parametrize("model", OFF_PEAK_MODELS)
@pytest.mark.parametrize("moment", (_WEEKDAY_OFF_PEAK, _WEEKEND), ids=("weekday-off-peak", "weekend"))
def test_apimodels_deepseek_rows_bill_half_rate_outside_weekday_peak_hours(model: str, moment: datetime):
    from litellm.cost_calculator import cost_per_token

    entry: Final = litellm.model_cost[model]
    with pinned_billing_time(moment):
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=1_000,
            completion_tokens=1_000,
            custom_llm_provider="apimodels",
        )

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] / 2 * 1_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] / 2 * 1_000)


def test_apimodels_off_peak_rows_are_the_deepseek_flash_models():
    assert OFF_PEAK_MODELS == ("apimodels/deepseek-v4-flash", "apimodels/deepseek-v4.1-flash")
