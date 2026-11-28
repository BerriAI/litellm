import json
import platform
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm._version import version as LITELLM_VERSION
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASE: Final = "https://api.openlux.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_USER_AGENT: Final = f"litellm/{LITELLM_VERSION} Python/{platform.python_version()}"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_openlux",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenLux", "reasoning_content": "Say hi"},
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
OPENLUX_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("openlux/")))
OPENLUX_CACHED_MODELS: Final = tuple(
    name for name in OPENLUX_MODELS if "cache_read_input_token_cost" in litellm.model_cost[name]
)


def _openlux_config() -> SimpleProviderConfig:
    config: Final = JSONProviderRegistry.get("openlux")
    assert config is not None
    return config


def _cost_of_one_million_in_and_out_with_400k_cache_hits(model: str) -> float:
    return litellm.completion_cost(
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
        custom_llm_provider="openlux",
    )


def test_openlux_is_a_registered_provider():
    assert litellm.LlmProviders.OPENLUX.value == "openlux"
    assert "openlux" in litellm.provider_list
    assert "openlux" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert OPENLUX_MODELS
    assert set(litellm.models_by_provider["openlux"]) == set(OPENLUX_MODELS)


def test_openlux_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "openlux-test-key")
    monkeypatch.delenv("OPENLUX_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="openlux/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-pro"
    assert provider == "openlux"
    assert api_key == "openlux-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_openlux_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "openlux-env-key")
    monkeypatch.setenv("OPENLUX_API_BASE", "https://openlux.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="openlux/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base="https://openlux.internal.example/v1",
        api_key="openlux-explicit-key",
    )

    assert provider == "openlux"
    assert api_key == "openlux-explicit-key"
    assert api_base == "https://openlux.internal.example/v1"


def test_openlux_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "openlux-env-key")
    monkeypatch.setenv("OPENLUX_API_BASE", "https://openlux.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="openlux/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "openlux"
    assert api_base == "https://openlux.env.example/v1"


def test_openlux_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "openlux-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "deepseek-v4-pro"
    assert provider == "openlux"
    assert api_key == "openlux-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", OPENLUX_CACHED_MODELS)
def test_openlux_bills_the_listed_rates_with_cache_hits(model: str):
    rates: Final = litellm.model_cost[model]
    expected: Final = (
        600_000 * rates["input_cost_per_token"]
        + 400_000 * rates["cache_read_input_token_cost"]
        + 1_000_000 * rates["output_cost_per_token"]
    )

    assert _cost_of_one_million_in_and_out_with_400k_cache_hits(model) == pytest.approx(expected)


@pytest.mark.usefixtures("local_model_cost_map")
def test_openlux_wildcard_expands_to_every_openlux_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("openlux/*")) == set(OPENLUX_MODELS)


def test_openlux_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("openlux/")}

    assert tuple(sorted(entries)) == OPENLUX_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_openlux_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    openlux: Final = next(provider for provider in providers if provider["litellm_provider"] == "openlux")

    assert openlux["provider"] == "OPENLUX"
    assert openlux["default_model_placeholder"] in OPENLUX_MODELS
    assert {field["key"]: field["required"] for field in openlux["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_openlux_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["openlux"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["openlux"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_openlux_attribution_user_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

    assert dict(_openlux_config().get_attribution_headers()) == {"User-Agent": _USER_AGENT}


def test_openlux_chat_completion_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="openlux/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            api_key="openlux-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer openlux-test-key"
    assert request.headers["User-Agent"] == _USER_AGENT
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "high"
    assert response.choices[0].message.content == "Hello from OpenLux"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert response.usage.prompt_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_openlux_caller_user_agent_wins():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).mock(
            return_value=httpx.Response(200, json=_CHAT_COMPLETION)
        )
        litellm.completion(
            model="openlux/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="openlux-test-key",
            headers={"User-Agent": "caller-app/1.0"},
        )

    assert route.calls.last.request.headers["User-Agent"] == "caller-app/1.0"


def test_openlux_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="openlux/deepseek-v4-pro",
            input="Say hello",
            api_key="openlux-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer openlux-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from OpenLux"


@pytest.mark.asyncio
async def test_openlux_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="openlux/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="openlux-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer openlux-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from OpenLux" for block in response["content"])
