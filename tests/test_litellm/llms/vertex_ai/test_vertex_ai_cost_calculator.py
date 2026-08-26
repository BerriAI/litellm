import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.types.utils import Usage


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


GEMINI_3X_VERTEX_SERVICE_TIER_RATES = [
    (None, 3e-07, 2.5e-06),
    ("flex", 1.5e-07, 1.25e-06),
    ("priority", 5.4e-07, 4.5e-06),
]


@pytest.mark.parametrize("service_tier,input_rate,output_rate", GEMINI_3X_VERTEX_SERVICE_TIER_RATES)
def test_vertex_gemini_3x_chat_honors_service_tier(service_tier, input_rate, output_rate, _local_model_cost_map):
    """Regression: cost_router sends vertex_ai gemini-3.x chat to cost_per_character, whose
    per-token fallback used to drop service_tier, so flex/priority-served responses billed at
    standard rates."""
    usage = Usage(prompt_tokens=4_061, completion_tokens=34, total_tokens=4_095)

    prompt_cost, completion_cost = cost_per_token(
        model="vertex_ai/gemini-3.5-flash-lite",
        custom_llm_provider="vertex_ai",
        usage_object=usage,
        service_tier=service_tier,
    )

    assert prompt_cost == pytest.approx(4_061 * input_rate, rel=1e-9)
    assert completion_cost == pytest.approx(34 * output_rate, rel=1e-9)
