from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.lens.inference import Deployment, DeploymentParams, completion_charge, quote
from litellm.types.utils import ModelResponse


def test_missing_optional_price_tiers_use_base_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-base-rate-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_above_200k_tokens": None,
                "output_cost_per_token_above_200k_tokens": None,
                "input_cost_per_token_above_128k_tokens": None,
                "output_cost_per_token_above_128k_tokens": None,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-base-rate-test"))
    explicit: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-base-rate-test", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
    )
    assert quote((deployment,), "Answer the question") == quote((explicit,), "Answer the question")


def test_unpriced_model_requires_explicit_rates() -> None:
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-unpriced-test"))
    with pytest.raises(HTTPException) as error:
        quote((deployment,), "Answer the question")
    assert error.value.status_code == 400
    assert "input_cost_per_token" in error.value.detail
    assert "output_cost_per_token" in error.value.detail


def test_custom_priced_model_charges_reported_tokens() -> None:
    deployment: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-test", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
    )
    response: Final = ModelResponse(
        model="lens-test", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    )
    assert completion_charge((deployment,), response, 10) == pytest.approx(0.04)
    assert quote((deployment,), "hello") > 0.04
