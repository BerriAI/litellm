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
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASE: Final = "https://proxystacks.com/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_RESPONSES_URL: Final = f"{_DEFAULT_API_BASE}/responses"
_MESSAGES_URL: Final = f"{_DEFAULT_API_BASE}/messages"

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_proxystacks",
    "object": "chat.completion",
    "created": 1_795_000_000,
    "model": "glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from ProxyStacks", "reasoning_content": "Think first"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}

_RESPONSES_PAYLOAD: Final = {
    "id": "resp_proxystacks",
    "object": "response",
    "created_at": 1_795_000_000,
    "status": "completed",
    "model": "gpt-6.1-sol",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Responses OK", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    },
}

_MESSAGES_PAYLOAD: Final = {
    "id": "msg_proxystacks",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Messages OK"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 3},
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

# DeepSeek peak hours on ProxyStacks Relay: Mon-Fri 01:00-03:59 and 06:00-09:59 UTC
# (https://proxystacks.com/public/model-rates, retrieved 2026-11-25)
_WEEKDAY_PEAK_MINUTE: Final = datetime(2026, 11, 25, 2, 30, tzinfo=timezone.utc)  # Wednesday 02:30 UTC
_WEEKDAY_OFF_PEAK_MINUTE: Final = datetime(2026, 11, 25, 5, 0, tzinfo=timezone.utc)  # Wednesday 05:00 UTC
_WEEKEND_OFF_PEAK_MINUTE: Final = datetime(2026, 11, 28, 2, 30, tzinfo=timezone.utc)  # Saturday 02:30 UTC

PROXYSTACKS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("proxystacks/")))


def test_proxystacks_is_a_registered_provider():
    assert litellm.LlmProviders.PROXYSTACKS.value == "proxystacks"
    assert "proxystacks" in litellm.provider_list
    assert "proxystacks" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert JSONProviderRegistry.exists("proxystacks")
    cfg: Final = JSONProviderRegistry.get("proxystacks")
    assert cfg is not None
    assert cfg.base_url == _DEFAULT_API_BASE
    assert cfg.api_key_env == "RELAY_API_KEY"
    assert cfg.api_base_env == "RELAY_API_BASE"
    assert list(cfg.supported_endpoints) == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
    assert len(PROXYSTACKS_MODELS) == 21
    assert set(litellm.models_by_provider["proxystacks"]) == set(PROXYSTACKS_MODELS)


def test_proxystacks_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELAY_API_KEY", "relay-test-key")
    monkeypatch.delenv("RELAY_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="proxystacks/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3"
    assert provider == "proxystacks"
    assert api_key == "relay-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_proxystacks_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELAY_API_KEY", "relay-env-key")
    monkeypatch.setenv("RELAY_API_BASE", "https://hk.proxystacks.com/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="proxystacks/glm-5.3",
        custom_llm_provider=None,
        api_base="https://custom.proxystacks.example/v1",
        api_key="relay-explicit-key",
    )

    assert provider == "proxystacks"
    assert api_key == "relay-explicit-key"
    assert api_base == "https://custom.proxystacks.example/v1"


def test_proxystacks_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELAY_API_KEY", "relay-env-key")
    monkeypatch.setenv("RELAY_API_BASE", "https://hk.proxystacks.com/v1")

    _, provider, _, api_base = get_llm_provider(
        model="proxystacks/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "proxystacks"
    assert api_base == "https://hk.proxystacks.com/v1"


def test_proxystacks_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELAY_API_KEY", "relay-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="glm-5.3",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "glm-5.3"
    assert provider == "proxystacks"
    assert api_key == "relay-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_proxystacks_deepseek_peak_and_off_peak_billing():
    model: Final = "proxystacks/deepseek-v4.1-flash"
    usage: Final = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400_000),
    )

    with pinned_billing_time(_WEEKDAY_PEAK_MINUTE):
        peak_cost = litellm.completion_cost(
            completion_response=ModelResponse(model=model, usage=usage),
            model=model,
            custom_llm_provider="proxystacks",
        )
    assert peak_cost == pytest.approx(600_000 * 1.8e-07 + 400_000 * 3.9e-09 + 1_000_000 * 7.4e-07)

    for moment in (_WEEKDAY_OFF_PEAK_MINUTE, _WEEKEND_OFF_PEAK_MINUTE):
        with pinned_billing_time(moment):
            off_peak_cost = litellm.completion_cost(
                completion_response=ModelResponse(model=model, usage=usage),
                model=model,
                custom_llm_provider="proxystacks",
            )
        assert off_peak_cost == pytest.approx(600_000 * 8.95e-08 + 400_000 * 2e-09 + 1_000_000 * 3.7e-07)


@pytest.mark.parametrize(
    ("model", "threshold", "base_in", "tier_in", "base_out", "tier_out"),
    (
        ("proxystacks/gpt-6.1-sol", 272_000, 1.4e-06, 2.8e-06, 7e-06, 1.05e-05),
        ("proxystacks/claude-haiku-5-5", 100_000, 5e-08, 2.5e-07, 2.5e-07, 1.25e-06),
    ),
)
def test_proxystacks_long_context_tier_billing(
    model: str,
    threshold: int,
    base_in: float,
    tier_in: float,
    base_out: float,
    tier_out: float,
):
    at_threshold = litellm.cost_per_token(
        model=model,
        custom_llm_provider="proxystacks",
        prompt_tokens=threshold,
        completion_tokens=1_000,
    )
    assert at_threshold == pytest.approx((threshold * base_in, 1_000 * base_out))

    above_threshold = litellm.cost_per_token(
        model=model,
        custom_llm_provider="proxystacks",
        prompt_tokens=threshold + 1,
        completion_tokens=1_000,
    )
    assert above_threshold == pytest.approx(((threshold + 1) * tier_in, 1_000 * tier_out))


@pytest.mark.usefixtures("local_model_cost_map")
def test_proxystacks_wildcard_expands_to_every_proxystacks_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("proxystacks/*")) == set(PROXYSTACKS_MODELS)


def test_proxystacks_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("proxystacks/")}

    assert tuple(sorted(entries)) == PROXYSTACKS_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_proxystacks_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    proxystacks: Final = next(provider for provider in providers if provider["litellm_provider"] == "proxystacks")

    assert proxystacks["provider"] == "PROXYSTACKS"
    assert proxystacks["default_model_placeholder"] in PROXYSTACKS_MODELS
    assert {field["key"]: field["required"] for field in proxystacks["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_proxystacks_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["proxystacks"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["proxystacks"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_proxystacks_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="proxystacks/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            api_key="relay-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer relay-test-key"
    assert body["model"] == "glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "high"
    assert response.choices[0].message.content == "Hello from ProxyStacks"
    assert response.choices[0].message.reasoning_content == "Think first"
    assert response.usage.prompt_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_proxystacks_native_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_RESPONSES_URL).respond(200, json=_RESPONSES_PAYLOAD)
        response: Final = litellm.responses(
            model="proxystacks/gpt-6.1-sol",
            input="Say hello",
            api_key="relay-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _RESPONSES_URL
    assert request.headers["authorization"] == "Bearer relay-test-key"
    assert body["model"] == "gpt-6.1-sol"
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Responses OK"


@pytest.mark.asyncio
async def test_proxystacks_native_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_MESSAGES_URL).respond(200, json=_MESSAGES_PAYLOAD)
        response: Final = await litellm.anthropic.messages.acreate(
            model="proxystacks/claude-sonnet-5-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Cached prompt",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                }
            ],
            max_tokens=32,
            api_key="relay-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _MESSAGES_URL
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 32
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert any(block.get("text") == "Messages OK" for block in response["content"])
