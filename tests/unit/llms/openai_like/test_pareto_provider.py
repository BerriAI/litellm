import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.constants import OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.types.utils import Usage

GLM: Final = "z-ai/glm-5.3-flash"
DEEPSEEK: Final = "deepseek/deepseek-v4.1-flash"
PARETO_MODELS: Final = tuple(sorted(f"pareto/{model}" for model in (GLM, DEEPSEEK)))
PACKAGE_ROOT: Final = Path(litellm.__file__).parent
COST_MAPS: Final = (
    PACKAGE_ROOT.parent / "model_prices_and_context_window.json",
    PACKAGE_ROOT / "model_prices_and_context_window_backup.json",
)
CHAT_COMPLETION: Final = {
    "id": "chatcmpl-pareto",
    "object": "chat.completion",
    "created": 1,
    "model": GLM,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
# Wednesday 2026-11-04: 02:00 UTC is a DeepSeek peak hour, 12:00 UTC is off-peak
PEAK: Final = datetime(2026, 11, 4, 2, tzinfo=timezone.utc)
OFF_PEAK: Final = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
SATURDAY: Final = datetime(2026, 11, 7, 2, tzinfo=timezone.utc)


@pytest.fixture
def pareto() -> SimpleProviderConfig:
    config: Final = JSONProviderRegistry.get("pareto")
    assert config is not None
    return config


def test_prefixed_model_resolves_to_pareto_and_keeps_the_vendor_namespace(
    pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARETO_API_BASE", raising=False)

    model, provider, _, api_base = get_llm_provider(model=f"pareto/{GLM}")

    assert provider == litellm.LlmProviders.PARETO.value
    assert model == GLM
    assert api_base == pareto.base_url


def test_api_key_and_base_are_read_from_pareto_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")
    monkeypatch.setenv("PARETO_API_BASE", "https://pareto.internal.example/v1")

    _, provider, api_key, api_base = get_llm_provider(model=f"pareto/{GLM}")

    assert provider == "pareto"
    assert api_key == "sk-pareto-env"
    assert api_base == "https://pareto.internal.example/v1"


def test_default_api_base_autodetects_pareto(pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")

    model, provider, api_key, _ = get_llm_provider(model=GLM, api_base=pareto.base_url)

    assert provider == "pareto"
    assert model == GLM
    assert api_key == "sk-pareto-env"


def test_pareto_is_not_an_audio_transcription_provider() -> None:
    assert "pareto" not in litellm.openai_compatible_providers
    assert "pareto" not in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS


@respx.mock
def test_completion_posts_to_pareto_chat_completions_with_bearer_key(
    pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARETO_API_BASE", raising=False)
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")
    route: Final = respx.post(f"{pareto.base_url}/chat/completions").respond(json=CHAT_COMPLETION)

    response: Final = litellm.completion(
        model=f"pareto/{GLM}",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
    )

    assert route.call_count == 1
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sk-pareto-env"
    assert body["model"] == GLM
    assert body["tools"][0]["function"]["name"] == "noop"
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 6


def test_responses_request_is_bridged_to_chat_completions(pareto: SimpleProviderConfig) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{pareto.base_url}/chat/completions").respond(json=CHAT_COMPLETION)
        response: Final = litellm.responses(model=f"pareto/{GLM}", input="hello", api_key="sk-pareto-test")

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == GLM
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "hi"


@pytest.mark.asyncio
async def test_anthropic_messages_request_is_bridged_to_chat_completions(
    pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{pareto.base_url}/chat/completions").respond(json=CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"pareto/{GLM}",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=32,
            api_key="sk-pareto-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == GLM
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "hi" for block in response["content"])


def test_cost_maps_carry_the_same_pareto_entries() -> None:
    main, backup = (
        {name: entry for name, entry in json.loads(path.read_text()).items() if name.startswith("pareto/")}
        for path in COST_MAPS
    )

    assert tuple(sorted(main)) == PARETO_MODELS
    assert main == backup
    assert {entry["litellm_provider"] for entry in main.values()} == {"pareto"}


@pytest.mark.usefixtures("local_model_cost_map")
def test_pareto_models_lists_every_priced_model() -> None:
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("pareto/*")) == set(PARETO_MODELS)


def test_pareto_is_available_in_add_model_form() -> None:
    fields_path: Final = PACKAGE_ROOT / "proxy" / "public_endpoints" / "provider_create_fields.json"
    entry: Final = next(
        provider for provider in json.loads(fields_path.read_text()) if provider["litellm_provider"] == "pareto"
    )

    assert entry["provider_display_name"] == "Pareto Inference"
    assert entry["default_model_placeholder"] in PARETO_MODELS
    assert {field["key"]: field["required"] for field in entry["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


@pytest.mark.usefixtures("local_model_cost_map")
def test_glm_bills_published_rates() -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=f"pareto/{GLM}", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )

    assert prompt_cost == pytest.approx(0.06)
    assert completion_cost == pytest.approx(0.20)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("when", "input_rate", "output_rate"),
    [(PEAK, 0.21, 0.84), (OFF_PEAK, 0.105, 0.42), (SATURDAY, 0.105, 0.42)],
)
def test_deepseek_bills_peak_and_off_peak_rates(when: datetime, input_rate: float, output_rate: float) -> None:
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

    prompt_cost, completion_cost = generic_cost_per_token(
        model=DEEPSEEK,
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
        custom_llm_provider="pareto",
        current_time=when,
    )

    assert prompt_cost == pytest.approx(input_rate)
    assert completion_cost == pytest.approx(output_rate)
