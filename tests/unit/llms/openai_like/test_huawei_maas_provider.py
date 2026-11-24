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
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASE: Final = "https://api-ap-southeast-1.modelarts-maas.com/openai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_huawei_maas",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Huawei", "reasoning_content": "Say hi"},
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
_LAST_PEAK_MINUTE: Final = datetime(2026, 11, 24, 12, 59, tzinfo=timezone.utc)
_FIRST_OFF_PEAK_MINUTE: Final = datetime(2026, 11, 24, 13, 0, tzinfo=timezone.utc)
_LAST_OFF_PEAK_MINUTE: Final = datetime(2026, 11, 24, 23, 59, tzinfo=timezone.utc)
_FIRST_PEAK_MINUTE: Final = datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)
HUAWEI_MAAS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("huawei_maas/")))
HUAWEI_MAAS_OFF_PEAK_MODELS: Final = tuple(
    name for name in HUAWEI_MAAS_MODELS if "off_peak_pricing" in litellm.model_cost[name]
)


def _cost_of_one_million_in_and_out_with_400k_cache_hits(model: str, moment: datetime) -> float:
    with pinned_billing_time(moment):
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
            custom_llm_provider="huawei_maas",
        )


def _expected_cost(rates: dict[str, float]) -> float:
    return (
        600_000 * rates["input_cost_per_token"]
        + 400_000 * rates["cache_read_input_token_cost"]
        + 1_000_000 * rates["output_cost_per_token"]
    )


def test_huawei_maas_is_a_registered_provider():
    assert litellm.LlmProviders.HUAWEI_MAAS.value == "huawei_maas"
    assert "huawei_maas" in litellm.provider_list
    assert "huawei_maas" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert HUAWEI_MAAS_MODELS
    assert set(litellm.models_by_provider["huawei_maas"]) == set(HUAWEI_MAAS_MODELS)


def test_huawei_maas_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUAWEI_MAAS_API_KEY", "huawei-test-key")
    monkeypatch.delenv("HUAWEI_MAAS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="huawei_maas/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3"
    assert provider == "huawei_maas"
    assert api_key == "huawei-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_huawei_maas_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUAWEI_MAAS_API_KEY", "huawei-env-key")
    monkeypatch.setenv("HUAWEI_MAAS_API_BASE", "https://huawei.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="huawei_maas/glm-5.3",
        custom_llm_provider=None,
        api_base="https://huawei.internal.example/v1",
        api_key="huawei-explicit-key",
    )

    assert provider == "huawei_maas"
    assert api_key == "huawei-explicit-key"
    assert api_base == "https://huawei.internal.example/v1"


def test_huawei_maas_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUAWEI_MAAS_API_KEY", "huawei-env-key")
    monkeypatch.setenv("HUAWEI_MAAS_API_BASE", "https://huawei.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="huawei_maas/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "huawei_maas"
    assert api_base == "https://huawei.env.example/v1"


def test_huawei_maas_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUAWEI_MAAS_API_KEY", "huawei-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="glm-5.3",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "glm-5.3"
    assert provider == "huawei_maas"
    assert api_key == "huawei-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", HUAWEI_MAAS_MODELS)
@pytest.mark.parametrize("moment", [_LAST_PEAK_MINUTE, _FIRST_PEAK_MINUTE], ids=["12:59", "00:00"])
def test_huawei_maas_bills_the_listed_rate_in_period_one(model: str, moment: datetime):
    assert _cost_of_one_million_in_and_out_with_400k_cache_hits(model, moment) == pytest.approx(
        _expected_cost(litellm.model_cost[model])
    )


@pytest.mark.parametrize("model", HUAWEI_MAAS_OFF_PEAK_MODELS)
@pytest.mark.parametrize("moment", [_FIRST_OFF_PEAK_MINUTE, _LAST_OFF_PEAK_MINUTE], ids=["13:00", "23:59"])
def test_huawei_maas_bills_the_off_peak_rate_in_period_two(model: str, moment: datetime):
    off_peak: Final = litellm.model_cost[model]["off_peak_pricing"]

    assert _expected_cost(off_peak) < _expected_cost(litellm.model_cost[model])
    assert _cost_of_one_million_in_and_out_with_400k_cache_hits(model, moment) == pytest.approx(
        _expected_cost(off_peak)
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_huawei_maas_wildcard_expands_to_every_huawei_maas_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("huawei_maas/*")) == set(HUAWEI_MAAS_MODELS)


def test_huawei_maas_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("huawei_maas/")}

    assert tuple(sorted(entries)) == HUAWEI_MAAS_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_huawei_maas_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    huawei: Final = next(provider for provider in providers if provider["litellm_provider"] == "huawei_maas")

    assert huawei["provider"] == "HUAWEI_MAAS"
    assert huawei["default_model_placeholder"] in HUAWEI_MAAS_MODELS
    assert {field["key"]: field["required"] for field in huawei["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_huawei_maas_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["huawei_maas"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["huawei_maas"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_huawei_maas_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="huawei_maas/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            api_key="huawei-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer huawei-test-key"
    assert body["model"] == "glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "high"
    assert response.choices[0].message.content == "Hello from Huawei"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert response.usage.prompt_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_huawei_maas_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="huawei_maas/glm-5.3",
            input="Say hello",
            api_key="huawei-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer huawei-test-key"
    assert body["model"] == "glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Huawei"


@pytest.mark.asyncio
async def test_huawei_maas_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="huawei_maas/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="huawei-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer huawei-test-key"
    assert body["model"] == "glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Huawei" for block in response["content"])
