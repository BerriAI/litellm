import math

import pytest

import litellm
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def test_stepfun_models_in_model_cost(local_model_cost_map):
    stepfun_models = [
        "stepfun/step-3.7-flash",
        "stepfun/step-3.5-flash",
        "stepfun/step-3.5-flash-2603",
    ]

    for model in stepfun_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "stepfun"


def test_stepfun_step_35_flash_cost_calculation(local_model_cost_map):
    prompt_cost, completion_cost = cost_per_token(
        model="stepfun/step-3.5-flash",
        prompt_tokens=1000000,
        completion_tokens=1000000,
        custom_llm_provider="stepfun",
    )

    assert math.isclose(prompt_cost, 0.10, rel_tol=1e-6)
    assert math.isclose(completion_cost, 0.30, rel_tol=1e-6)


def test_stepfun_step_37_flash_cost_calculation(local_model_cost_map):
    prompt_cost, completion_cost = cost_per_token(
        model="stepfun/step-3.7-flash",
        prompt_tokens=1000000,
        completion_tokens=1000000,
        custom_llm_provider="stepfun",
    )

    assert math.isclose(prompt_cost, 0.20, rel_tol=1e-6)
    assert math.isclose(completion_cost, 1.15, rel_tol=1e-6)


def test_stepfun_step_35_flash_cache_hit_cost(local_model_cost_map):
    prompt_cost, _ = cost_per_token(
        model="stepfun/step-3.5-flash",
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_input_tokens=1000000,
        custom_llm_provider="stepfun",
    )

    assert math.isclose(prompt_cost, 0.02, rel_tol=1e-6)


def test_stepfun_reasoning_models_support_reasoning(local_model_cost_map):
    for model in ("stepfun/step-3.5-flash", "stepfun/step-3.5-flash-2603", "stepfun/step-3.7-flash"):
        assert litellm.model_cost[model]["supports_reasoning"] is True
