import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASE: Final = "https://api.hiapi.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_hiapi",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from HiAPI", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
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
HIAPI_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("hiapi/")))


def test_hiapi_is_a_registered_provider():
    assert litellm.LlmProviders.HIAPI.value == "hiapi"
    assert "hiapi" in litellm.provider_list
    assert "hiapi" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert HIAPI_MODELS
    assert set(litellm.models_by_provider["hiapi"]) == set(HIAPI_MODELS)


def test_hiapi_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIAPI_API_KEY", "hiapi-test-key")
    monkeypatch.delenv("HIAPI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="hiapi/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-pro"
    assert provider == "hiapi"
    assert api_key == "hiapi-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_hiapi_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIAPI_API_KEY", "hiapi-env-key")
    monkeypatch.setenv("HIAPI_API_BASE", "https://hiapi.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="hiapi/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base="https://hiapi.internal.example/v1",
        api_key="hiapi-explicit-key",
    )

    assert provider == "hiapi"
    assert api_key == "hiapi-explicit-key"
    assert api_base == "https://hiapi.internal.example/v1"


def test_hiapi_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIAPI_API_KEY", "hiapi-env-key")
    monkeypatch.setenv("HIAPI_API_BASE", "https://hiapi.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="hiapi/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "hiapi"
    assert api_base == "https://hiapi.env.example/v1"


def test_hiapi_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIAPI_API_KEY", "hiapi-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "deepseek-v4-pro"
    assert provider == "hiapi"
    assert api_key == "hiapi-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_hiapi_does_not_send_an_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HIAPI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-that-must-stay-home")

    _, provider, api_key, _ = get_llm_provider(
        model="hiapi/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "hiapi"
    assert api_key != "openai-key-that-must-stay-home"


@pytest.mark.parametrize("model", HIAPI_MODELS)
def test_hiapi_bills_the_listed_rates(model: str):
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
        custom_llm_provider="hiapi",
    )

    expected: Final = (
        600_000 * rates["input_cost_per_token"]
        + 400_000 * rates.get("cache_read_input_token_cost", rates["input_cost_per_token"])
        + 1_000_000 * rates["output_cost_per_token"]
    )
    assert cost == pytest.approx(expected)


@pytest.mark.usefixtures("local_model_cost_map")
def test_hiapi_wildcard_expands_to_every_hiapi_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("hiapi/*")) == set(HIAPI_MODELS)


def test_hiapi_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("hiapi/")}

    assert tuple(sorted(entries)) == HIAPI_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_hiapi_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    hiapi: Final = next(provider for provider in providers if provider["litellm_provider"] == "hiapi")

    assert hiapi["provider"] == "HIAPI"
    assert hiapi["default_model_placeholder"] in HIAPI_MODELS
    assert {field["key"]: field["required"] for field in hiapi["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_hiapi_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["hiapi"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["hiapi"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_hiapi_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="hiapi/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            api_key="hiapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer hiapi-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "high"
    assert response.choices[0].message.content == "Hello from HiAPI"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert response.usage.prompt_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_hiapi_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="hiapi/deepseek-v4-pro",
            input="Say hello",
            api_key="hiapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer hiapi-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from HiAPI"


@pytest.mark.asyncio
async def test_hiapi_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="hiapi/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="hiapi-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer hiapi-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from HiAPI" for block in response["content"])
