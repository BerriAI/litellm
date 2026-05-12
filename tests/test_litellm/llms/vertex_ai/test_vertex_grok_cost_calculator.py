import os

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map():
    previous_value = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    previous_model_cost = litellm.model_cost
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    yield
    if previous_value is None:
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous_value
    litellm.model_cost = previous_model_cost


@pytest.mark.parametrize(
    "model,prompt_tokens,cached_tokens,text_tokens,completion_tokens,reasoning_tokens,input_cost,cache_read_cost,output_cost,expected_cost",
    [
        (
            "vertex_ai/xai/grok-4.1-fast-non-reasoning",
            1000,
            250,
            750,
            100,
            0,
            2e-07,
            5e-08,
            5e-07,
            (750 * 2e-07) + (250 * 5e-08) + (100 * 5e-07),
        ),
        (
            "vertex_ai/xai/grok-4.20-reasoning",
            1000,
            400,
            600,
            200,
            50,
            2e-06,
            2e-07,
            6e-06,
            (600 * 2e-06) + (400 * 2e-07) + (200 * 6e-06),
        ),
    ],
)
def test_vertex_grok_cost_calculation_with_cached_and_reasoning_tokens(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    text_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    input_cost: float,
    cache_read_cost: float,
    output_cost: float,
    expected_cost: float,
):
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")
    assert model_info["input_cost_per_token"] == input_cost
    assert model_info["cache_read_input_token_cost"] == cache_read_cost
    assert model_info["output_cost_per_token"] == output_cost

    completion_tokens_details = None
    if reasoning_tokens:
        completion_tokens_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=reasoning_tokens,
            text_tokens=completion_tokens - reasoning_tokens,
        )

    response = ModelResponse(
        id="vertex-grok-cost-test",
        model=model,
        choices=[],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=cached_tokens,
                text_tokens=text_tokens,
            ),
            completion_tokens_details=completion_tokens_details,
        ),
    )

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="vertex_ai",
    )

    assert pytest.approx(cost, rel=1e-6) == expected_cost
