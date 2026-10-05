from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.lens.inference import Deployment, DeploymentParams, completion_charge, model_step, quote
from litellm.proxy.lens.models import ModelRequest
from litellm.types.utils import ModelResponse


def test_missing_optional_price_tiers_use_base_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-base-rate-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 16384,
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
            model="openai/lens-base-rate-test",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            max_tokens=16384,
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
            model="openai/lens-test", input_cost_per_token=0.001, output_cost_per_token=0.002, max_tokens=16384
        )
    )
    response: Final = ModelResponse(
        model="lens-test", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    )
    assert completion_charge((deployment,), response, 10) == pytest.approx(0.04)
    assert quote((deployment,), "hello") > 0.04


@pytest.mark.parametrize("capacity", (8192, 65536, 128000))
def test_output_allowance_and_budget_follow_the_models_capacity(capacity: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.lens.inference import output_tokens

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-capacity-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": capacity,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0.001,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-capacity-test"))
    assert output_tokens(deployment) == capacity
    assert quote((deployment,), "Review") == pytest.approx(capacity * 0.001)


def test_explicit_deployment_output_setting_is_respected() -> None:
    from litellm.proxy.lens.inference import output_tokens

    deployment: Final = Deployment(litellm_params=DeploymentParams(model="custom/model", max_tokens=32000))
    assert output_tokens(deployment) == 32000


def test_shared_context_capacity_leaves_room_for_the_entire_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.lens.inference import output_tokens

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-shared-context": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 8192,
                "max_input_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0.001,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-shared-context"))
    short: Final = output_tokens(deployment, "Review this trace")
    long: Final = output_tokens(deployment, "Review this trace " * 500)
    assert 0 < long < short < output_tokens(deployment)
    assert quote((deployment,), "Review this trace " * 500) == pytest.approx(long * 0.001)


def test_unknown_model_capacity_requires_explicit_operator_metadata() -> None:
    from litellm.proxy.lens.inference import ModelCapacity, output_tokens

    params: Final = DeploymentParams(model="openai/lens-unknown-capacity")
    with pytest.raises(HTTPException) as error:
        output_tokens(Deployment(litellm_params=params))
    assert error.value.status_code == 400
    assert "model_info.max_output_tokens" in error.value.detail
    configured: Final = Deployment(litellm_params=params, model_info=ModelCapacity(max_output_tokens=32000))
    assert output_tokens(configured) == 32000


def test_a_model_step_records_the_serving_model_and_its_tokens() -> None:
    response: Final = ModelResponse(model="gpt-5.6", usage={"prompt_tokens": 1200, "completion_tokens": 80})
    step: Final = model_step(response, ModelRequest(prompt="review", purpose="extract"), "analysis", 0.02)
    assert (step.model, step.prompt_tokens, step.completion_tokens, step.cost) == ("gpt-5.6", 1200, 80, 0.02)


def test_a_response_without_usage_still_records_a_step_instead_of_failing_settlement() -> None:
    response: Final = ModelResponse(model="gpt-5.6")
    unpriced: Final = response.model_copy(update={"usage": None})
    step: Final = model_step(unpriced, ModelRequest(prompt="review", purpose="cluster"), "analysis", 0.0)
    assert (step.prompt_tokens, step.completion_tokens) == (0, 0)
    assert step.label == "Compared observations"
