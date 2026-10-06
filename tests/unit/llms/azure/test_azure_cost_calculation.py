"""
Test Azure OpenAI cost calculator — service_tier pricing.
"""

from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.cost_calculation import cost_per_token
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from litellm.utils import get_model_info


# Register a test model with tier-specific pricing
TEST_MODEL = "test-azure-gpt-4.1"
TEST_MODEL_COST = {
    TEST_MODEL: {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_priority": 0.01,
        "output_cost_per_token_priority": 0.02,
        "input_cost_per_token_flex": 0.0005,
        "output_cost_per_token_flex": 0.001,
        "litellm_provider": "azure",
        "max_tokens": 8192,
    }
}


class TestAzureServiceTierCostCalculation:
    """Test that service_tier is passed through Azure cost calculation."""

    @pytest.fixture(autouse=True)
    def register_test_model(self):
        litellm.register_model(model_cost=TEST_MODEL_COST)

    def test_service_tier_priority_higher_cost(self):
        """Priority tier should cost more than standard."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage
        )
        priority_prompt, priority_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="priority"
        )

        assert priority_prompt > standard_prompt
        assert priority_completion > standard_completion

    def test_service_tier_flex_lower_cost(self):
        """Flex tier should cost less than standard."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage
        )
        flex_prompt, flex_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="flex"
        )

        assert flex_prompt < standard_prompt
        assert flex_completion < standard_completion

    def test_service_tier_none_returns_standard(self):
        """service_tier=None should return standard pricing."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        none_prompt, none_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier=None
        )
        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="standard"
        )

        assert abs(none_prompt - standard_prompt) < 1e-10
        assert abs(none_completion - standard_completion) < 1e-10


ROUTED_MODEL: Final = "gpt-5-nano-2025-08-07"
ROUTED_USAGE: Final = Usage(prompt_tokens=5000, completion_tokens=2000, total_tokens=7000)


def _router_fee() -> float:
    fee_per_token: Final = get_model_info(model="model_router", custom_llm_provider="azure_ai")["input_cost_per_token"]
    return ROUTED_USAGE.prompt_tokens * (fee_per_token or 0.0)


def _azure_logging(request_model: str) -> Logging:
    return Logging(
        model=request_model,
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="test-azure-model-router",
        function_id="test-function",
    )


def _azure_response(response_model: str) -> ModelResponse:
    response: Final = ModelResponse(
        id="test-azure-model-router",
        choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="Hello"))],
        created=1234567890,
        model=response_model,
        object="chat.completion",
        usage=ROUTED_USAGE,
    )
    response._hidden_params = {"custom_llm_provider": "azure"}
    return response


def _routed_model_cost() -> tuple[float, float]:
    routed_info: Final = get_model_info(model=ROUTED_MODEL, custom_llm_provider="azure")
    return (
        ROUTED_USAGE.prompt_tokens * (routed_info["input_cost_per_token"] or 0.0),
        ROUTED_USAGE.completion_tokens * (routed_info["output_cost_per_token"] or 0.0),
    )


@pytest.mark.usefixtures("local_model_cost_map")
class TestAzureModelRouterCostBreakdown:
    """azure/ charges the Model Router fee once through the same additional cost line as azure_ai/."""

    def test_router_request_with_routed_response_adds_the_fee(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        logging_obj = _azure_logging("azure-model-router")
        cost = completion_cost(
            completion_response=_azure_response(ROUTED_MODEL),
            model=ROUTED_MODEL,
            custom_llm_provider="azure",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert breakdown["input_cost"] == pytest.approx(routed_prompt_cost, rel=1e-9)
        assert breakdown["output_cost"] == pytest.approx(routed_completion_cost, rel=1e-9)
        assert breakdown.get("additional_costs") == pytest.approx(
            {"Azure Model Router Flat Cost": _router_fee()}, rel=1e-9
        )
        assert cost == pytest.approx(routed_prompt_cost + routed_completion_cost + _router_fee(), rel=1e-9)

    def test_normal_azure_call_is_priced_as_the_model_alone(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        logging_obj = _azure_logging("gpt-5-nano")
        cost = completion_cost(
            completion_response=_azure_response(ROUTED_MODEL),
            model=ROUTED_MODEL,
            custom_llm_provider="azure",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert "additional_costs" not in breakdown
        assert cost == pytest.approx(routed_prompt_cost + routed_completion_cost, rel=1e-9)

    def test_response_priced_as_the_router_entry_charges_the_fee_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "azure/model-router",
            {**litellm.model_cost["azure_ai/model-router"], "litellm_provider": "azure"},
        )
        litellm.get_model_info.cache_clear()
        logging_obj = _azure_logging("model-router")
        cost = completion_cost(
            completion_response=_azure_response("model-router"),
            model="model-router",
            custom_llm_provider="azure",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert "additional_costs" not in breakdown
        assert breakdown["input_cost"] == pytest.approx(_router_fee(), rel=1e-9)
        assert cost == pytest.approx(_router_fee(), rel=1e-9)
