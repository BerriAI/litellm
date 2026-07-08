import pytest

import litellm
from litellm.llms.tencent.cost_calculator import cost_per_token
from litellm.types.utils import Usage


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


def test_cost_per_token_uses_tencent_model_pricing(local_model_cost_map):
    usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    prompt_cost, completion_cost = cost_per_token(model="tencent/deepseek-v4-pro", usage=usage)

    assert prompt_cost == pytest.approx(1000 * 4.35e-07)
    assert completion_cost == pytest.approx(2000 * 8.7e-07)


def test_top_level_dispatcher_routes_tencent_to_wrapper(local_model_cost_map):
    from litellm.cost_calculator import cost_per_token as dispatch_cost_per_token

    prompt_cost, completion_cost = dispatch_cost_per_token(
        model="tencent/deepseek-v4-pro",
        prompt_tokens=1000,
        completion_tokens=1000,
        custom_llm_provider="tencent",
    )

    assert prompt_cost == pytest.approx(1000 * 4.35e-07)
    assert completion_cost == pytest.approx(1000 * 8.7e-07)
