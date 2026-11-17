from typing import Final

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


@pytest.mark.parametrize("cache_rate", [0.0, 3e-8])
@pytest.mark.parametrize("cached_tokens", [0, 75200])
@pytest.mark.parametrize("custom_pricing", [False, True])
def test_together_completion_cost_preserves_registered_rates(
    monkeypatch: pytest.MonkeyPatch,
    cache_rate: float,
    cached_tokens: int,
    custom_pricing: bool,
) -> None:
    deployment_id: Final = "00000000-3b00-4000-8000-000000000001"
    model: Final = "example/pricing-regression-3B"
    registry_key: Final = deployment_id if custom_pricing else f"together_ai/{model}"
    input_rate: Final = 1.5e-7
    output_rate: Final = 5e-7
    prompt_tokens: Final = 89536
    completion_tokens: Final = 2117
    monkeypatch.setitem(
        litellm.model_cost,
        "together-ai-up-to-4b",
        {"input_cost_per_token": 9e-6, "output_cost_per_token": 9e-6, "litellm_provider": "together_ai"},
    )
    litellm.register_model(
        model_cost={
            registry_key: {
                "input_cost_per_token": input_rate,
                "output_cost_per_token": output_rate,
                "cache_read_input_token_cost": cache_rate,
                "litellm_provider": "together_ai",
            }
        }
    )
    response: Final = ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
        ),
    )

    cost: Final = completion_cost(
        completion_response=response,
        custom_llm_provider="together_ai",
        custom_pricing=custom_pricing,
        router_model_id=deployment_id,
    )

    assert cost == pytest.approx(
        (prompt_tokens - cached_tokens) * input_rate + cached_tokens * cache_rate + completion_tokens * output_rate
    )


def test_together_completion_cost_keeps_unregistered_model_size_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    input_rate: Final = 2e-6
    output_rate: Final = 7e-6
    monkeypatch.setitem(
        litellm.model_cost,
        "together-ai-up-to-4b",
        {"input_cost_per_token": input_rate, "output_cost_per_token": output_rate, "litellm_provider": "together_ai"},
    )
    response: Final = ModelResponse(
        model="example/unregistered-3B",
        usage=Usage(prompt_tokens=100, completion_tokens=20),
    )

    assert completion_cost(
        completion_response=response,
        custom_llm_provider="together_ai",
    ) == pytest.approx(100 * input_rate + 20 * output_rate)


@pytest.mark.parametrize(
    ("bare_provider", "declares_provider"),
    [("openai", True), (None, False), (None, True), ("together_ai", True)],
)
@pytest.mark.parametrize("provider_scoped", [False, True])
def test_together_completion_cost_ignores_other_provider_bare_prices(
    monkeypatch: pytest.MonkeyPatch,
    bare_provider: str | None,
    declares_provider: bool,
    provider_scoped: bool,
) -> None:
    model: Final = "example/colliding-3B"
    fallback_rates: Final = (2e-6, 7e-6)
    bare_rates: Final = (3e-6, 11e-6)
    scoped_rates: Final = (5e-6, 13e-6)
    monkeypatch.setitem(
        litellm.model_cost,
        "together-ai-up-to-4b",
        {
            "input_cost_per_token": fallback_rates[0],
            "output_cost_per_token": fallback_rates[1],
            "litellm_provider": "together_ai",
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": bare_rates[0],
            "output_cost_per_token": bare_rates[1],
            **({"litellm_provider": bare_provider} if declares_provider else {}),
        },
    )
    if provider_scoped:
        monkeypatch.setitem(
            litellm.model_cost,
            f"together_ai/{model}",
            {
                "input_cost_per_token": scoped_rates[0],
                "output_cost_per_token": scoped_rates[1],
                "litellm_provider": "together_ai",
            },
        )
    response: Final = ModelResponse(model=model, usage=Usage(prompt_tokens=100, completion_tokens=20))
    input_rate, output_rate = (
        scoped_rates if provider_scoped else bare_rates if bare_provider in (None, "together_ai") else fallback_rates
    )

    assert completion_cost(
        completion_response=response,
        custom_llm_provider="together_ai",
    ) == pytest.approx(100 * input_rate + 20 * output_rate)
