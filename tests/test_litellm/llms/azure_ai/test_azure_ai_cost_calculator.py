"""
Test Azure AI cost calculator, especially Model Router flat cost.
"""

from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure_ai.cost_calculator import (
    calculate_azure_model_router_flat_cost,
    cost_per_token,
    is_azure_model_router,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from litellm.utils import get_model_info

# Get the flat cost from model_prices_and_context_window.json
_model_info = get_model_info(model="model_router", custom_llm_provider="azure_ai")
AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS = _model_info.get("input_cost_per_token", 0) * 1_000_000


class TestAzureModelRouterDetection:
    """Test that we correctly identify Azure Model Router models.

    Model Router deployments follow the pattern: model_router/<deployment-name>
    where deployment-name is the Azure deployment (e.g., 'azure-model-router', 'prod-router')
    """

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Deployment names containing 'model-router' or 'model_router'
            ("azure-model-router", True),
            ("AZURE-MODEL-ROUTER", True),
            ("model-router", True),
            ("MODEL-ROUTER", True),
            ("my-model-router-deployment", True),
            ("prod-model_router", True),
            # New pattern: model_router/<deployment-name>
            ("model_router/azure-model-router", True),
            ("model-router/prod-router", True),
            ("model_router/my-deployment", True),
            ("MODEL_ROUTER/AZURE-MODEL-ROUTER", True),
            # Non-router models
            ("gpt-4o", False),
            ("gpt-4o-mini", False),
            ("claude-sonnet-4-5", False),
            ("my-regular-deployment", False),
        ],
    )
    def test_is_azure_model_router(self, model: str, expected: bool):
        """Test Azure Model Router detection."""
        assert is_azure_model_router(model) == expected


class TestAzureModelRouterPrefix:
    """Test Azure Model Router prefix stripping."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Model router deployments - the deployment name comes after model_router/
            ("model_router/azure-model-router", "azure-model-router"),
            ("model-router/my-router-deployment", "my-router-deployment"),
            ("model_router/prod-router", "prod-router"),
            # Non-router models - should pass through unchanged
            ("gpt-4o", "gpt-4o"),
            ("azure-model-router", "azure-model-router"),
            ("claude-sonnet-4", "claude-sonnet-4"),
        ],
    )
    def test_strip_model_router_prefix(self, model: str, expected: str):
        """Test that model_router prefix is stripped correctly.

        The pattern is: model_router/<deployment-name>
        where deployment-name is the Azure deployment (e.g., 'azure-model-router', 'prod-router')
        """
        from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

        result = AzureFoundryModelInfo.strip_model_router_prefix(model)
        assert result == expected


ROUTER_FEE_PER_TOKEN: Final = AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
ROUTED_MODEL: Final = "gpt-4.1-nano-2025-04-14"
ROUTED_USAGE: Final = Usage(prompt_tokens=5000, completion_tokens=2000, total_tokens=7000)
ROUTED_FEE: Final = 5000 * ROUTER_FEE_PER_TOKEN


def _router_logging(request_model: str) -> Logging:
    return Logging(
        model=request_model,
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="test-123",
        function_id="test-function",
    )


def _azure_ai_response(response_model: str, litellm_model_name: str | None = None) -> ModelResponse:
    response: Final = ModelResponse(
        id="test-123",
        choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="Hello"))],
        created=1234567890,
        model=response_model,
        object="chat.completion",
        usage=ROUTED_USAGE,
    )
    response._hidden_params = (
        {"custom_llm_provider": "azure_ai"}
        if litellm_model_name is None
        else {"custom_llm_provider": "azure_ai", "litellm_model_name": litellm_model_name}
    )
    return response


def _routed_model_cost() -> tuple[float, float]:
    routed_info: Final = get_model_info(model=ROUTED_MODEL, custom_llm_provider="azure_ai")
    return (
        ROUTED_USAGE.prompt_tokens * (routed_info["input_cost_per_token"] or 0.0),
        ROUTED_USAGE.completion_tokens * (routed_info["output_cost_per_token"] or 0.0),
    )


@pytest.mark.usefixtures("local_model_cost_map")
class TestAzureModelRouterFlatCost:
    """cost_per_token charges the router fee once, for whichever router name the caller gives it."""

    def test_unmapped_router_deployment_name_prices_the_fee(self) -> None:
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        prompt_cost, completion_cost_usd = cost_per_token(model="azure-model-router", usage=usage)
        assert prompt_cost == pytest.approx(1000 * ROUTER_FEE_PER_TOKEN, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_unmapped_router_deployment_name_charges_the_fee_over_cached_prompt_tokens_too(self) -> None:
        usage = Usage(
            prompt_tokens=2000,
            completion_tokens=800,
            total_tokens=2800,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=200,
        )
        prompt_cost, completion_cost_usd = cost_per_token(model="azure-model-router", usage=usage)
        assert prompt_cost == pytest.approx(2000 * ROUTER_FEE_PER_TOKEN, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_router_deployment_name_as_both_names_charges_the_fee_once(self) -> None:
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        prompt_cost, completion_cost_usd = cost_per_token(
            model="model_router/my-deployment", usage=usage, request_model="azure_ai/model_router/my-deployment"
        )
        assert prompt_cost == pytest.approx(1000 * ROUTER_FEE_PER_TOKEN, rel=1e-9)
        assert completion_cost_usd == 0.0

    @pytest.mark.parametrize("router_entry_name", ["model_router", "model-router"])
    def test_router_entry_prices_its_own_fee(self, router_entry_name: str) -> None:
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000)
        prompt_cost, completion_cost_usd = cost_per_token(model=router_entry_name, usage=usage)
        assert prompt_cost == pytest.approx(0.14, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_routed_model_is_priced_as_itself(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        prompt_cost, completion_cost_usd = cost_per_token(model=ROUTED_MODEL, usage=ROUTED_USAGE)
        assert routed_prompt_cost > 0
        assert prompt_cost == pytest.approx(routed_prompt_cost, rel=1e-9)
        assert completion_cost_usd == pytest.approx(routed_completion_cost, rel=1e-9)

    def test_unmapped_model_that_is_not_a_router_name_raises(self) -> None:
        usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        with pytest.raises(Exception, match="no-such-azure-ai-model"):
            cost_per_token(model="no-such-azure-ai-model", usage=usage)

    def test_request_model_through_the_router_adds_the_fee_once(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        prompt_cost, completion_cost_usd = cost_per_token(
            model=ROUTED_MODEL, usage=ROUTED_USAGE, request_model="azure_ai/model-router"
        )
        assert prompt_cost == pytest.approx(routed_prompt_cost + ROUTED_FEE, rel=1e-9)
        assert completion_cost_usd == pytest.approx(routed_completion_cost, rel=1e-9)

    def test_request_model_that_is_not_the_router_adds_nothing(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        assert cost_per_token(
            model=ROUTED_MODEL, usage=ROUTED_USAGE, request_model=f"azure_ai/{ROUTED_MODEL}"
        ) == pytest.approx((routed_prompt_cost, routed_completion_cost), rel=1e-9)

    @pytest.mark.parametrize("router_entry_name", ["model_router", "model-router"])
    def test_request_model_does_not_double_the_router_entry(self, router_entry_name: str) -> None:
        prompt_cost, completion_cost_usd = cost_per_token(
            model=router_entry_name, usage=ROUTED_USAGE, request_model=f"azure_ai/{router_entry_name}"
        )
        assert prompt_cost == pytest.approx(ROUTED_FEE, rel=1e-9)
        assert completion_cost_usd == 0.0

    def test_public_cost_per_token_keeps_the_request_model_keyword(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        prompt_cost, completion_cost_usd = litellm.cost_per_token(
            model=ROUTED_MODEL,
            custom_llm_provider="azure_ai",
            usage_object=ROUTED_USAGE,
            request_model="azure_ai/model-router",
        )
        assert prompt_cost == pytest.approx(routed_prompt_cost + ROUTED_FEE, rel=1e-9)
        assert completion_cost_usd == pytest.approx(routed_completion_cost, rel=1e-9)

    def test_flat_cost_helper(self) -> None:
        assert calculate_azure_model_router_flat_cost(
            model="azure-model-router", prompt_tokens=10_000
        ) == pytest.approx(0.0014, rel=1e-9)
        assert calculate_azure_model_router_flat_cost(model="gpt-5-nano", prompt_tokens=10_000) == 0.0

    def test_flat_cost_reads_the_fee_from_the_deployment_named_entry(self) -> None:
        litellm.register_model(
            {"azure_ai/model-router": {"input_cost_per_token": 2e-07, "litellm_provider": "azure_ai", "mode": "chat"}}
        )
        litellm.get_model_info.cache_clear()
        assert calculate_azure_model_router_flat_cost(model="model-router", prompt_tokens=1_000_000) == pytest.approx(
            0.2, rel=1e-9
        )
        assert calculate_azure_model_router_flat_cost(
            model="azure-model-router", prompt_tokens=1_000_000
        ) == pytest.approx(0.14, rel=1e-9)


@pytest.mark.usefixtures("local_model_cost_map")
class TestAzureModelRouterCostBreakdown:
    """completion_cost charges the router fee exactly once: as the breakdown's additional cost line when a routed
    model is priced as itself, inside the input cost when the priced name is the router."""

    def test_unmapped_router_deployment_name_costs_only_the_fee(self) -> None:
        cost = completion_cost(
            completion_response=_azure_ai_response("azure-model-router"),
            model="azure-model-router",
            custom_llm_provider="azure_ai",
        )
        assert cost == pytest.approx(ROUTED_FEE, rel=1e-9)

    def test_unmapped_router_name_carries_the_fee_as_its_input_cost(self) -> None:
        logging_obj = _router_logging("azure-model-router")
        cost = completion_cost(
            completion_response=_azure_ai_response("azure-model-router"),
            model="azure-model-router",
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert breakdown["input_cost"] == pytest.approx(ROUTED_FEE, rel=1e-9)
        assert "additional_costs" not in breakdown
        assert cost == pytest.approx(ROUTED_FEE, rel=1e-9)

    def test_router_request_with_routed_response_charges_the_fee_once(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        logging_obj = _router_logging("model-router")
        cost = completion_cost(
            completion_response=_azure_ai_response(ROUTED_MODEL),
            model=ROUTED_MODEL,
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert breakdown["input_cost"] == pytest.approx(routed_prompt_cost, rel=1e-9)
        assert breakdown["output_cost"] == pytest.approx(routed_completion_cost, rel=1e-9)
        assert breakdown.get("additional_costs") == pytest.approx(
            {"Azure Model Router Flat Cost": ROUTED_FEE}, rel=1e-9
        )
        assert cost == pytest.approx(routed_prompt_cost + routed_completion_cost + ROUTED_FEE, rel=1e-9)

    def test_routed_response_named_by_hidden_params_charges_the_fee_once(self) -> None:
        routed_prompt_cost, routed_completion_cost = _routed_model_cost()
        logging_obj = _router_logging(ROUTED_MODEL)
        cost = completion_cost(
            completion_response=_azure_ai_response(ROUTED_MODEL, litellm_model_name="azure_ai/model-router"),
            model=ROUTED_MODEL,
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert breakdown["input_cost"] == pytest.approx(routed_prompt_cost, rel=1e-9)
        assert breakdown.get("additional_costs") == pytest.approx(
            {"Azure Model Router Flat Cost": ROUTED_FEE}, rel=1e-9
        )
        assert cost == pytest.approx(routed_prompt_cost + routed_completion_cost + ROUTED_FEE, rel=1e-9)

    @pytest.mark.parametrize("router_entry_name", ["model_router", "model-router"])
    def test_response_priced_as_the_router_entry_charges_the_fee_once(self, router_entry_name: str) -> None:
        logging_obj = _router_logging(router_entry_name)
        cost = completion_cost(
            completion_response=_azure_ai_response(router_entry_name),
            model=router_entry_name,
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )
        breakdown = logging_obj.cost_breakdown
        assert breakdown is not None
        assert "additional_costs" not in breakdown
        assert breakdown["input_cost"] == pytest.approx(ROUTED_FEE, rel=1e-9)
        assert cost == pytest.approx(ROUTED_FEE, rel=1e-9)


class TestAzureAIServiceTierCostCalculation:
    """Test that service_tier is passed through Azure AI cost calculation."""

    @pytest.fixture(autouse=True)
    def register_test_model(self):
        import litellm

        litellm.register_model(
            model_cost={
                "test-azure-ai-model": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    "input_cost_per_token_priority": 0.01,
                    "output_cost_per_token_priority": 0.02,
                    "input_cost_per_token_flex": 0.0005,
                    "output_cost_per_token_flex": 0.001,
                    "litellm_provider": "azure_ai",
                    "max_tokens": 8192,
                }
            }
        )

    def test_service_tier_priority_higher_cost(self):
        """Priority tier should cost more than standard for azure_ai."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(model="test-azure-ai-model", usage=usage)
        priority_prompt, priority_completion = cost_per_token(
            model="test-azure-ai-model", usage=usage, service_tier="priority"
        )

        assert priority_prompt > standard_prompt
        assert priority_completion > standard_completion

    def test_service_tier_flex_lower_cost(self):
        """Flex tier should cost less than standard for azure_ai."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(model="test-azure-ai-model", usage=usage)
        flex_prompt, flex_completion = cost_per_token(model="test-azure-ai-model", usage=usage, service_tier="flex")

        assert flex_prompt < standard_prompt
        assert flex_completion < standard_completion


def test_codestral_2501_model_info_and_cost(local_model_cost_map):
    model_info = get_model_info(model="Codestral-2501", custom_llm_provider="azure_ai")
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    prompt_cost, completion_cost = cost_per_token(model="Codestral-2501", usage=usage)

    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 256000
    assert model_info["max_output_tokens"] == 4096
    assert prompt_cost == pytest.approx(0.3)
    assert completion_cost == pytest.approx(0.9)


def test_mai_thinking_1_model_info_and_cost(local_model_cost_map):
    model_info = get_model_info(model="MAI-Thinking-1", custom_llm_provider="azure_ai")
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    prompt_cost, completion_cost = cost_per_token(model="MAI-Thinking-1", usage=usage)

    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 256000
    assert model_info["max_output_tokens"] == 64000
    assert model_info["cache_read_input_token_cost"] == pytest.approx(2e-07)
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_function_calling"] is True
    assert prompt_cost == pytest.approx(2.0)
    assert completion_cost == pytest.approx(8.0)
