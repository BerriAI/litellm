import pytest

import litellm
from litellm.llms.melious.cost_calculator import cost_per_token
from litellm.types.utils import Usage

GLM_51_INPUT_COST = 1.3e-06
GLM_51_OUTPUT_COST = 4.05e-06


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_cost_per_token_uses_melious_model_pricing(local_model_cost_map):
    usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    prompt_cost, completion_cost = cost_per_token(model="melious/glm-5.1", usage=usage)

    assert prompt_cost == pytest.approx(1000 * GLM_51_INPUT_COST)
    assert completion_cost == pytest.approx(2000 * GLM_51_OUTPUT_COST)


def test_top_level_dispatcher_routes_melious_to_wrapper(local_model_cost_map):
    from litellm.cost_calculator import cost_per_token as dispatch_cost_per_token

    prompt_cost, completion_cost = dispatch_cost_per_token(
        model="melious/glm-5.1",
        prompt_tokens=1000,
        completion_tokens=1000,
        custom_llm_provider="melious",
    )

    assert prompt_cost == pytest.approx(1000 * GLM_51_INPUT_COST)
    assert completion_cost == pytest.approx(1000 * GLM_51_OUTPUT_COST)


def test_melious_models_are_registered_under_the_provider(local_model_cost_map):
    litellm.add_known_models()

    assert "melious/glm-5.1" in litellm.models_by_provider["melious"]
    assert litellm.get_model_info(model="melious/glm-5.1")["litellm_provider"] == "melious"
