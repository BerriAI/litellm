import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

BASE_URL: Final = "https://api-inference.bitdeer.ai/v1"
MODEL: Final = "moonshotai/Kimi-K3"
MODELS: Final = (MODEL, "zai-org/GLM-5.2", "deepseek-ai/DeepSeek-V4-Flash")


@pytest.mark.parametrize("api_key", (None, "explicit-key"))
@pytest.mark.parametrize("api_base", (BASE_URL, BASE_URL + "/"))
@pytest.mark.parametrize("routing", ("prefix", "provider", "url"))
def test_bitdeer_credentials(
    monkeypatch: pytest.MonkeyPatch, api_key: str | None, api_base: str, routing: str
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")

    result: Final = get_llm_provider(
        model=f"bitdeer-ai/{MODEL}" if routing == "prefix" else MODEL,
        custom_llm_provider="bitdeer-ai" if routing == "provider" else None,
        api_base=api_base,
        api_key=api_key,
    )

    assert result == (MODEL, "bitdeer-ai", api_key or "environment-key", api_base)


def test_bitdeer_default_base_and_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    assert get_llm_provider(model=f"bitdeer-ai/{MODEL}") == (MODEL, "bitdeer-ai", None, BASE_URL)
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    assert get_llm_provider(model=f"bitdeer-ai/{MODEL}") == (MODEL, "bitdeer-ai", "environment-key", BASE_URL)


@pytest.mark.parametrize("api_key", (None, ""))
@pytest.mark.parametrize(
    "api_base",
    (
        "https://custom.example/v1",
        "http://api-inference.bitdeer.ai/v1",
        "https://api-inference.bitdeer.ai:8443/v1",
        "https://api-inference.bitdeer.ai.evil.example/v1",
        "https://api-inference.bitdeer.ai@evil.example/v1",
        "https://evil.example/https://api-inference.bitdeer.ai/v1",
        BASE_URL + "/other",
        BASE_URL + "?redirect=https://evil.example",
    ),
)
def test_bitdeer_custom_base_requires_explicit_key(
    monkeypatch: pytest.MonkeyPatch, api_base: str, api_key: str | None
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")

    with pytest.raises(litellm.BadRequestError, match="explicit api_key is required") as error:
        get_llm_provider(model=f"bitdeer-ai/{MODEL}", api_base=api_base, api_key=api_key)

    assert "server-secret" not in str(error.value)


@pytest.mark.parametrize(
    "api_base", ("http://api-inference.bitdeer.ai/v1", "https://api-inference.bitdeer.ai:8443/v1", BASE_URL + "/other")
)
def test_bitdeer_url_autodetection_rejects_unsafe_key_fallback(
    monkeypatch: pytest.MonkeyPatch, api_base: str
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    with pytest.raises(litellm.BadRequestError, match="explicit api_key is required"):
        get_llm_provider(model=MODEL, api_base=api_base)


def test_bitdeer_dynamic_config_checks_custom_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    provider: Final = JSONProviderRegistry.get("bitdeer-ai")
    assert provider is not None
    config: Final = create_config_class(provider)()

    with pytest.raises(ValueError, match="explicit api_key is required"):
        config._get_openai_compatible_provider_info("https://custom.example/v1", None)

    assert config._get_openai_compatible_provider_info("https://custom.example/v1", "explicit-key") == (
        "https://custom.example/v1", "explicit-key"
    )


@pytest.mark.parametrize("model", (MODEL, f"bitdeer-ai/{MODEL}"))
@pytest.mark.parametrize("api_key", (None, "explicit-key"))
def test_bitdeer_completion_sends_selected_key(
    monkeypatch: pytest.MonkeyPatch, model: str, api_key: str | None
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(BASE_URL + "/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-bitdeer",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
        response: Final = litellm.completion(
            model=model,
            api_base=BASE_URL,
            api_key=api_key,
            messages=[{"role": "user", "content": "Hi"}],
            max_completion_tokens=32,
        )

    request: Final = route.calls.last.request
    payload: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == f"Bearer {api_key or 'environment-key'}"
    assert payload["model"] == MODEL
    assert payload["max_tokens"] == 32
    assert "max_completion_tokens" not in payload
    assert response.choices[0].message.content == "Hello"
    assert response._hidden_params["response_cost"] == pytest.approx(
        10 * litellm.model_cost[f"bitdeer-ai/{MODEL}"]["input_cost_per_token"]
        + 2 * litellm.model_cost[f"bitdeer-ai/{MODEL}"]["output_cost_per_token"]
    )


def test_bitdeer_completion_rejects_custom_base_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    with respx.mock() as upstream:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key is required"):
            litellm.completion(
                model=f"bitdeer-ai/{MODEL}",
                api_base="https://custom.example/v1",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert len(upstream.calls) == 0


def _unique_entries(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys: Final = tuple(key for key, _ in pairs)
    assert len(keys) == len(set(keys)), "Duplicate JSON key"
    return dict(pairs)


def test_bitdeer_pricing_and_packaged_registry() -> None:
    root: Final = Path(litellm.__file__).parent.parent
    prices: Final = json.loads((root / "model_prices_and_context_window.json").read_text(), object_pairs_hook=_unique_entries)
    backup: Final = json.loads((root / "litellm/model_prices_and_context_window_backup.json").read_text())

    for model in MODELS:
        name: Final = f"bitdeer-ai/{model}"
        entry: Final = prices[name]
        assert entry == backup[name] == litellm.model_cost[name]
        assert entry["max_tokens"] == entry["max_output_tokens"] <= entry["max_input_tokens"]
        assert entry["litellm_provider"] == "bitdeer-ai"
        prompt_cost, output_cost = litellm.cost_per_token(model=name, prompt_tokens=100, completion_tokens=10)
        assert prompt_cost == pytest.approx(100 * entry["input_cost_per_token"])
        assert output_cost == pytest.approx(10 * entry["output_cost_per_token"])
