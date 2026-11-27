import json
import re
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import PromptTokensDetailsWrapper

_DEFAULT_API_BASE: Final = "https://api.tokenless.store/api/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_UNLISTED_MODEL: Final = "tokenless/a-model-not-in-the-cost-map"
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
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_TIER_KEY: Final = re.compile(r"^input_cost_per_token_above_(\d+)k_tokens$")
TOKENLESS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("tokenless/")))


def _long_prompt_tiers(model: str) -> tuple[tuple[str, int], ...]:
    matches: Final = map(_TIER_KEY.match, litellm.model_cost[model])
    return tuple((model, int(match.group(1))) for match in matches if match is not None)


TOKENLESS_LONG_PROMPT_TIERS: Final = tuple(chain.from_iterable(map(_long_prompt_tiers, TOKENLESS_MODELS)))


def _chat_completion(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_tokenless",
        "object": "chat.completion",
        "created": 1_795_000_000,
        "model": "fable-5.2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Tokenless", "reasoning_content": "Say hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


_CHAT_COMPLETION: Final = _chat_completion(
    {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "prompt_tokens_details": {"cached_tokens": 4}}
)


def _cost_with_a_quarter_of_the_prompt_cached(model: str, prompt_tokens: int) -> float:
    return litellm.completion_cost(
        completion_response=litellm.ModelResponse(
            model=model.removeprefix("tokenless/"),
            usage=litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=200,
                total_tokens=prompt_tokens + 200,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=prompt_tokens // 4),
            ),
        ),
        model=model,
        custom_llm_provider="tokenless",
    )


def _expected_cost(model: str, prompt_tokens: int, tier: str = "") -> float:
    row: Final = litellm.model_cost[model]
    cached: Final = prompt_tokens // 4
    return (
        (prompt_tokens - cached) * row[f"input_cost_per_token{tier}"]
        + cached * row[f"cache_read_input_token_cost{tier}"]
        + 200 * row[f"output_cost_per_token{tier}"]
    )


def test_tokenless_is_a_registered_provider():
    assert litellm.LlmProviders.TOKENLESS.value == "tokenless"
    assert "tokenless" in litellm.provider_list
    assert "tokenless" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert TOKENLESS_MODELS
    assert set(litellm.models_by_provider["tokenless"]) == set(TOKENLESS_MODELS)


def test_tokenless_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENLESS_API_KEY", "sk-tk-test-key")
    monkeypatch.delenv("TOKENLESS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="tokenless/fable-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "fable-5.2"
    assert provider == "tokenless"
    assert api_key == "sk-tk-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_tokenless_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENLESS_API_KEY", "sk-tk-env-key")
    monkeypatch.setenv("TOKENLESS_API_BASE", "https://tokenless.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="tokenless/fable-5.2",
        custom_llm_provider=None,
        api_base="https://tokenless.internal.example/v1",
        api_key="sk-tk-explicit-key",
    )

    assert provider == "tokenless"
    assert api_key == "sk-tk-explicit-key"
    assert api_base == "https://tokenless.internal.example/v1"


def test_tokenless_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENLESS_API_KEY", "sk-tk-env-key")
    monkeypatch.setenv("TOKENLESS_API_BASE", "https://tokenless.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="tokenless/fable-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "tokenless"
    assert api_base == "https://tokenless.env.example/v1"


def test_tokenless_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENLESS_API_KEY", "sk-tk-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="fable-5.2",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "fable-5.2"
    assert provider == "tokenless"
    assert api_key == "sk-tk-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", TOKENLESS_MODELS)
def test_tokenless_row_fits_its_context_and_never_charges_more_for_cached_input(model: str):
    row: Final = litellm.model_cost[model]

    assert row["mode"] == "chat"
    assert 0 < row["max_output_tokens"] <= row["max_input_tokens"]
    assert 0 < row["cache_read_input_token_cost"] <= row["input_cost_per_token"]
    assert row["output_cost_per_token"] > 0


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", TOKENLESS_MODELS)
def test_tokenless_response_without_a_reported_cost_is_priced_at_its_row(model: str):
    assert _cost_with_a_quarter_of_the_prompt_cached(model, 1000) == pytest.approx(_expected_cost(model, 1000))


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "threshold_k"), TOKENLESS_LONG_PROMPT_TIERS)
def test_tokenless_prompt_at_the_tier_threshold_keeps_the_base_rate(model: str, threshold_k: int):
    prompt_tokens: Final = threshold_k * 1000

    assert _cost_with_a_quarter_of_the_prompt_cached(model, prompt_tokens) == pytest.approx(
        _expected_cost(model, prompt_tokens)
    )


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "threshold_k"), TOKENLESS_LONG_PROMPT_TIERS)
def test_tokenless_prompt_over_the_tier_threshold_is_priced_at_the_tier(model: str, threshold_k: int):
    prompt_tokens: Final = threshold_k * 1000 + 1
    tier: Final = f"_above_{threshold_k}k_tokens"

    assert _expected_cost(model, prompt_tokens, tier) > _expected_cost(model, prompt_tokens)
    assert _cost_with_a_quarter_of_the_prompt_cached(model, prompt_tokens) == pytest.approx(
        _expected_cost(model, prompt_tokens, tier)
    )


@pytest.mark.parametrize("model", TOKENLESS_MODELS)
def test_tokenless_model_capabilities_follow_its_row(model: str):
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_tool_choice,
        supports_vision,
    )

    row: Final = litellm.model_cost[model]
    model_info: Final = litellm.get_model_info(model)

    assert model_info["litellm_provider"] == "tokenless"
    assert model_info["max_tokens"] == row["max_output_tokens"]
    assert supports_function_calling(model) is row["supports_function_calling"]
    assert supports_tool_choice(model) is row["supports_tool_choice"]
    assert supports_reasoning(model) is row["supports_reasoning"]
    assert supports_prompt_caching(model) is row["supports_prompt_caching"]
    assert supports_vision(model) is row["supports_vision"]


def test_tokenless_model_missing_from_cost_map_supports_tools_and_reasoning():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling(_UNLISTED_MODEL) is True
    # Tokenless accepts reasoning_effort on every model, so an unlisted one must not reject it.
    assert supports_reasoning(_UNLISTED_MODEL) is True


@pytest.mark.usefixtures("local_model_cost_map")
def test_tokenless_wildcard_expands_to_every_tokenless_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("tokenless/*")) == set(TOKENLESS_MODELS)


def test_tokenless_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("tokenless/")}

    assert tuple(sorted(entries)) == TOKENLESS_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_tokenless_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    tokenless: Final = next(provider for provider in providers if provider["litellm_provider"] == "tokenless")

    assert tokenless["provider"] == "TOKENLESS"
    assert tokenless["default_model_placeholder"] in TOKENLESS_MODELS
    assert {field["key"]: field["required"] for field in tokenless["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_tokenless_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["tokenless"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["tokenless"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_tokenless_chat_completion_request_is_sent_as_the_caller_wrote_it():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="tokenless/fable-5.2",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            api_key="sk-tk-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer sk-tk-test-key"
    assert body["model"] == "fable-5.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "high"
    assert body["tools"] == [_WEATHER_TOOL]
    assert body["tool_choice"] == "auto"
    assert response.choices[0].message.content == "Hello from Tokenless"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert response.usage.prompt_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 4


@pytest.mark.parametrize("model", ("tokenless/fable-5.2", _UNLISTED_MODEL))
def test_tokenless_response_is_billed_at_the_cost_tokenless_reports(model: str):
    reported_cost: Final = 0.00721
    usage: Final = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": reported_cost}
    with respx.mock() as upstream:
        upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_chat_completion(usage))
        response: Final = litellm.completion(
            model=model, messages=[{"role": "user", "content": "Say hello"}], api_key="sk-tk-test-key"
        )

    assert response._hidden_params["response_cost"] == pytest.approx(reported_cost)


def test_tokenless_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="tokenless/gpt-6.2-astra",
            input="Say hello",
            api_key="sk-tk-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sk-tk-test-key"
    assert body["model"] == "gpt-6.2-astra"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Tokenless"


@pytest.mark.asyncio
async def test_tokenless_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="tokenless/fable-5.2",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-tk-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sk-tk-test-key"
    assert body["model"] == "fable-5.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Tokenless" for block in response["content"])
