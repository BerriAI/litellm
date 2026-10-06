"""
Test Azure OpenAI cost calculator: service_tier pricing and the Model Router fee.
"""

from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.cost_calculation import cost_per_token
from litellm.llms.azure_ai.cost_calculator import calculate_azure_model_router_flat_cost
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

        standard_prompt, standard_completion = cost_per_token(model=TEST_MODEL, usage=usage)
        priority_prompt, priority_completion = cost_per_token(model=TEST_MODEL, usage=usage, service_tier="priority")

        assert priority_prompt > standard_prompt
        assert priority_completion > standard_completion

    def test_service_tier_flex_lower_cost(self):
        """Flex tier should cost less than standard."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(model=TEST_MODEL, usage=usage)
        flex_prompt, flex_completion = cost_per_token(model=TEST_MODEL, usage=usage, service_tier="flex")

        assert flex_prompt < standard_prompt
        assert flex_completion < standard_completion

    def test_service_tier_none_returns_standard(self):
        """service_tier=None should return standard pricing."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        none_prompt, none_completion = cost_per_token(model=TEST_MODEL, usage=usage, service_tier=None)
        standard_prompt, standard_completion = cost_per_token(model=TEST_MODEL, usage=usage, service_tier="standard")

        assert abs(none_prompt - standard_prompt) < 1e-10
        assert abs(none_completion - standard_completion) < 1e-10


ROUTED_MODEL: Final = "gpt-5-nano-2025-08-07"
ROUTED_USAGE: Final = Usage(prompt_tokens=5000, completion_tokens=2000, total_tokens=7000)
AZURE_ROUTER_FEE_KEYS: Final = ("azure/model-router", "azure/model_router")


def _azure_ai_router_fee_per_token() -> float:
    return get_model_info(model="model_router", custom_llm_provider="azure_ai")["input_cost_per_token"] or 0.0


def _routed_model_cost() -> tuple[float, float]:
    routed_info: Final = get_model_info(model=ROUTED_MODEL, custom_llm_provider="azure")
    return (
        ROUTED_USAGE.prompt_tokens * (routed_info["input_cost_per_token"] or 0.0),
        ROUTED_USAGE.completion_tokens * (routed_info["output_cost_per_token"] or 0.0),
    )


def _cost_map_with_azure_router_fee(monkeypatch: pytest.MonkeyPatch, fee_per_token: float | None) -> None:
    azure_ai_entry: Final = litellm.model_cost["azure_ai/model-router"]
    cost_map: Final = {key: value for key, value in litellm.model_cost.items() if key not in AZURE_ROUTER_FEE_KEYS}
    azure_rows: Final = (
        {}
        if fee_per_token is None
        else {
            "azure/model-router": {
                **azure_ai_entry,
                "input_cost_per_token": fee_per_token,
                "litellm_provider": "azure",
            }
        }
    )
    monkeypatch.setattr(litellm, "model_cost", {**cost_map, **azure_rows})
    litellm.get_model_info.cache_clear()


@pytest.mark.usefixtures("local_model_cost_map")
class TestAzureModelRouterFee:
    """azure/ adds the Model Router fee once on top of the routed model, like azure_ai/."""

    def test_router_request_adds_the_fee_to_the_routed_model(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        fee: Final = ROUTED_USAGE.prompt_tokens * _azure_ai_router_fee_per_token()
        prompt_cost, completion_cost_usd = cost_per_token(
            model=ROUTED_MODEL, usage=ROUTED_USAGE, request_model="model-router"
        )
        assert routed_prompt_cost > 0 and fee > 0
        assert prompt_cost == pytest.approx(routed_prompt_cost + fee, rel=1e-9)
        assert completion_cost_usd == pytest.approx(routed_completion_cost, rel=1e-9)

    @pytest.mark.parametrize("router_entry_name", ["model-router", "model_router", "azure/model-router"])
    def test_priced_router_entry_charges_the_fee_once(self, router_entry_name: str) -> None:
        fee: Final = ROUTED_USAGE.prompt_tokens * _azure_ai_router_fee_per_token()
        prompt_cost, completion_cost_usd = cost_per_token(
            model=router_entry_name, usage=ROUTED_USAGE, request_model=router_entry_name
        )
        assert prompt_cost == pytest.approx(fee, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_priced_azure_router_row_charges_its_own_fee_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        azure_fee_per_token: Final = _azure_ai_router_fee_per_token() * 3
        _cost_map_with_azure_router_fee(monkeypatch, azure_fee_per_token)
        prompt_cost, completion_cost_usd = cost_per_token(
            model="model-router", usage=ROUTED_USAGE, request_model="model-router"
        )
        assert prompt_cost == pytest.approx(ROUTED_USAGE.prompt_tokens * azure_fee_per_token, rel=1e-9)
        assert completion_cost_usd == 0.0

    @pytest.mark.parametrize("request_model", [None, ROUTED_MODEL, f"azure/{ROUTED_MODEL}"])
    def test_non_router_request_is_priced_as_the_model_alone(self, request_model: str | None) -> None:
        assert cost_per_token(model=ROUTED_MODEL, usage=ROUTED_USAGE, request_model=request_model) == pytest.approx(
            _routed_model_cost(), rel=1e-9
        )

    def test_unmapped_model_without_router_request_still_raises(self) -> None:
        with pytest.raises(Exception, match="no-such-azure-model"):
            cost_per_token(model="no-such-azure-model", usage=ROUTED_USAGE)

    def test_router_request_with_unmapped_routed_model_costs_the_fee_alone(self) -> None:
        fee: Final = ROUTED_USAGE.prompt_tokens * _azure_ai_router_fee_per_token()
        prompt_cost, completion_cost_usd = cost_per_token(
            model="no-such-azure-model", usage=ROUTED_USAGE, request_model="my-model-router-deployment"
        )
        assert prompt_cost == pytest.approx(fee, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_fee_falls_back_to_the_azure_ai_entry_without_an_azure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fee_per_token: Final = _azure_ai_router_fee_per_token()
        _cost_map_with_azure_router_fee(monkeypatch, None)
        assert calculate_azure_model_router_flat_cost("model-router", 1000, custom_llm_provider="azure") == (
            pytest.approx(1000 * fee_per_token, rel=1e-9)
        )

    @pytest.mark.parametrize("router_name", ["model-router", "model_router", "azure-model-router"])
    def test_fee_prefers_the_azure_row_when_present(self, monkeypatch: pytest.MonkeyPatch, router_name: str) -> None:
        azure_fee_per_token: Final = _azure_ai_router_fee_per_token() * 2
        _cost_map_with_azure_router_fee(monkeypatch, azure_fee_per_token)
        assert calculate_azure_model_router_flat_cost(router_name, 1000, custom_llm_provider="azure") == (
            pytest.approx(1000 * azure_fee_per_token, rel=1e-9)
        )
        assert calculate_azure_model_router_flat_cost("model-router", 1000) == pytest.approx(
            1000 * _azure_ai_router_fee_per_token(), rel=1e-9
        )

    def test_completion_cost_adds_the_fee_for_a_router_request(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        fee: Final = ROUTED_USAGE.prompt_tokens * _azure_ai_router_fee_per_token()
        logging_obj: Final = Logging(
            model="model-router",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-azure-model-router",
            function_id="test-function",
        )
        response: Final = ModelResponse(
            id="test-azure-model-router",
            choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="Hello"))],
            created=1234567890,
            model=ROUTED_MODEL,
            object="chat.completion",
            usage=ROUTED_USAGE,
        )
        response._hidden_params = {"custom_llm_provider": "azure"}
        cost: Final = completion_cost(
            completion_response=response,
            model=ROUTED_MODEL,
            custom_llm_provider="azure",
            litellm_logging_obj=logging_obj,
        )
        breakdown: Final = logging_obj.cost_breakdown
        assert breakdown is not None
        assert "additional_costs" not in breakdown
        assert breakdown["input_cost"] == pytest.approx(routed_prompt_cost + fee, rel=1e-9)
        assert breakdown["output_cost"] == pytest.approx(routed_completion_cost, rel=1e-9)
        assert cost == pytest.approx(routed_prompt_cost + routed_completion_cost + fee, rel=1e-9)
