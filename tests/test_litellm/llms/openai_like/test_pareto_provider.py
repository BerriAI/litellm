import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig

MODEL: Final = "z-ai/glm-5.3-flash"
REPO_ROOT: Final = Path(__file__).parents[4]
COST_MAPS: Final = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)


@pytest.fixture
def pareto() -> SimpleProviderConfig:
    config: Final = JSONProviderRegistry.get("pareto")
    assert config is not None
    return config


def test_prefixed_model_resolves_to_pareto_and_keeps_the_vendor_namespace(
    pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARETO_API_BASE", raising=False)

    model, provider, _, api_base = get_llm_provider(model=f"pareto/{MODEL}")

    assert provider == litellm.LlmProviders.PARETO.value
    assert model == MODEL
    assert api_base == pareto.base_url


def test_api_key_and_base_are_read_from_pareto_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")
    monkeypatch.setenv("PARETO_API_BASE", "https://pareto.internal.example/v1")

    _, provider, api_key, api_base = get_llm_provider(model=f"pareto/{MODEL}")

    assert provider == "pareto"
    assert api_key == "sk-pareto-env"
    assert api_base == "https://pareto.internal.example/v1"


def test_default_api_base_autodetects_pareto(pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")

    model, provider, api_key, _ = get_llm_provider(model=MODEL, api_base=pareto.base_url)

    assert provider == "pareto"
    assert model == MODEL
    assert api_key == "sk-pareto-env"


@respx.mock
def test_completion_posts_to_pareto_chat_completions_with_bearer_key(
    pareto: SimpleProviderConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARETO_API_BASE", raising=False)
    monkeypatch.setenv("PARETO_API_KEY", "sk-pareto-env")
    route: Final = respx.post(f"{pareto.base_url}/chat/completions").respond(
        json={
            "id": "chatcmpl-pareto",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
    )

    response: Final = litellm.completion(
        model=f"pareto/{MODEL}",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-pareto-env"
    assert json.loads(request.content)["model"] == MODEL
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 6


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    """Resolve pricing against the in-repo cost map, not the remote one fetched at import."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def test_cost_maps_carry_the_same_pareto_entry() -> None:
    main, backup = (json.loads(path.read_text())[f"pareto/{MODEL}"] for path in COST_MAPS)

    assert main == backup
    assert main["litellm_provider"] == "pareto"


def test_published_per_token_prices_are_used_for_cost(local_model_cost_map: None) -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=f"pareto/{MODEL}", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )

    assert prompt_cost == pytest.approx(0.03)
    assert completion_cost == pytest.approx(0.10)
