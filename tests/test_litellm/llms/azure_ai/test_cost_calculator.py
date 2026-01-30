"""
Test Azure AI cost calculator, especially Model Router flat cost.
"""

import pytest
from litellm.llms.azure_ai.cost_calculator import (
    _is_azure_model_router,
    cost_per_token,
)
from litellm.types.utils import Usage
from litellm.utils import get_model_info

# Get the flat cost from model_prices_and_context_window.json
_model_info = get_model_info(model="azure-model-router", custom_llm_provider="azure_ai")
AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS = _model_info.get("input_cost_per_token", 0) * 1_000_000


class TestAzureModelRouterDetection:
    """Test that we correctly identify Azure Model Router models."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("azure-model-router", True),
            ("AZURE-MODEL-ROUTER", True),
            ("model-router", True),
            ("MODEL-ROUTER", True),
            ("gpt-4o-mini-2024-07-18-model-router", True),
            ("gpt-4o", False),
            ("gpt-4o-mini", False),
            ("claude-sonnet-4-5", False),
        ],
    )
    def test_is_azure_model_router(self, model: str, expected: bool):
        """Test Azure Model Router detection."""
        assert _is_azure_model_router(model) == expected


class TestAzureModelRouterFlatCost:
    """Test Azure AI Foundry Model Router flat cost calculation."""

    def test_model_router_flat_cost_basic(self):
        """Test that flat cost is added for Model Router requests."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # Calculate expected flat cost
        expected_flat_cost = (
            usage.prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        # Flat cost should be $0.00014 (1000 tokens × $0.14 / 1M tokens)
        assert expected_flat_cost == pytest.approx(0.00014, rel=1e-9)

        # Prompt cost should include the flat cost
        # (plus any base cost from the actual model used, which might be 0 if not in model_cost)
        assert prompt_cost >= expected_flat_cost
        print(
            f"Model Router flat cost for {usage.prompt_tokens} tokens: ${expected_flat_cost:.6f}"
        )
        print(f"Total prompt cost: ${prompt_cost:.6f}")

    def test_model_router_flat_cost_large_request(self):
        """Test flat cost calculation for larger requests."""
        model = "model-router"
        usage = Usage(
            prompt_tokens=100_000,
            completion_tokens=50_000,
            total_tokens=150_000,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # Calculate expected flat cost
        expected_flat_cost = (
            usage.prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        # Flat cost should be $0.014 (100k tokens × $0.14 / 1M tokens)
        assert expected_flat_cost == pytest.approx(0.014, rel=1e-9)
        assert prompt_cost >= expected_flat_cost
        print(
            f"Model Router flat cost for {usage.prompt_tokens} tokens: ${expected_flat_cost:.6f}"
        )
        print(f"Total prompt cost: ${prompt_cost:.6f}")

    def test_model_router_flat_cost_1m_tokens(self):
        """Test flat cost for exactly 1 million input tokens."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            total_tokens=1_100_000,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # Calculate expected flat cost
        expected_flat_cost = AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS

        # Flat cost should be exactly $0.14 for 1M tokens
        assert expected_flat_cost == pytest.approx(0.14, rel=1e-9)
        assert prompt_cost >= expected_flat_cost
        print(f"Model Router flat cost for 1M tokens: ${expected_flat_cost:.6f}")
        print(f"Total prompt cost: ${prompt_cost:.6f}")

    def test_non_model_router_no_flat_cost(self):
        """Test that non-Model Router models don't get the flat cost."""
        model = "gpt-4o"
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # No flat cost should be added for non-Model Router models
        # The cost might be 0 or based on the model's pricing
        print(f"Non-Model Router prompt cost: ${prompt_cost:.6f}")
        # We just ensure it doesn't crash and returns valid values
        assert prompt_cost >= 0
        assert completion_cost >= 0

    def test_model_router_with_cached_tokens(self):
        """Test Model Router flat cost with cached tokens."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=2000,
            completion_tokens=800,
            total_tokens=2800,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=200,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # Flat cost is based on ALL prompt tokens (including cached)
        expected_flat_cost = (
            usage.prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        assert expected_flat_cost == pytest.approx(0.00028, rel=1e-9)
        assert prompt_cost >= expected_flat_cost
        print(
            f"Model Router flat cost with caching for {usage.prompt_tokens} tokens: ${expected_flat_cost:.6f}"
        )
        print(f"Total prompt cost: ${prompt_cost:.6f}")


class TestAzureModelRouterCostBreakdown:
    """Test that Azure Model Router flat cost is tracked in cost breakdown."""

    def test_flat_cost_tracked_in_thread_local(self):
        """Test that flat cost is stored in thread-local storage."""
        from litellm.llms.azure_ai.cost_calculator import (
            get_azure_model_router_flat_cost,
        )

        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
        )

        # Calculate cost (this should store flat cost in thread-local)
        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # Retrieve the flat cost from thread-local storage
        flat_cost = get_azure_model_router_flat_cost()

        # Expected flat cost
        expected_flat_cost = (
            usage.prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        assert flat_cost is not None
        assert flat_cost == pytest.approx(expected_flat_cost, rel=1e-9)
        print(f"Flat cost tracked in thread-local: ${flat_cost:.6f}")

    def test_flat_cost_integration_with_completion_cost(self):
        """Test that flat cost is properly integrated into completion_cost calculation."""
        import litellm
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import Usage, ModelResponse, Choices, Message

        # Create a mock response for azure_ai model router
        response = ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Test response",
                    ),
                )
            ],
            created=1234567890,
            model="azure-model-router",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=5000,
                completion_tokens=2000,
                total_tokens=7000,
            ),
        )

        # Set hidden params for provider
        response._hidden_params = {"custom_llm_provider": "azure_ai"}

        # Calculate cost
        cost = completion_cost(
            completion_response=response,
            model="azure-model-router",
            custom_llm_provider="azure_ai",
        )

        # Expected flat cost
        expected_flat_cost = (
            5000 * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        # Cost should include the flat cost
        assert cost > expected_flat_cost
        print(f"Total cost with flat fee: ${cost:.6f}")
        print(f"Expected minimum flat cost: ${expected_flat_cost:.6f}")

    def test_additional_costs_in_cost_breakdown(self):
        """Test that Azure Model Router flat cost appears in additional_costs dict."""
        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import LitellmLoggingObject
        from litellm.types.utils import Usage, ModelResponse, Choices, Message

        # Create logging object
        logging_obj = LitellmLoggingObject()

        # Create a mock response for azure_ai model router
        response = ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Test response",
                    ),
                )
            ],
            created=1234567890,
            model="azure-model-router",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=5000,
                completion_tokens=2000,
                total_tokens=7000,
            ),
        )

        # Set hidden params for provider
        response._hidden_params = {"custom_llm_provider": "azure_ai"}

        # Calculate cost with logging object
        cost = completion_cost(
            completion_response=response,
            model="azure-model-router",
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )

        # Check that cost breakdown contains additional_costs
        assert hasattr(logging_obj, "cost_breakdown")
        assert logging_obj.cost_breakdown is not None
        assert "additional_costs" in logging_obj.cost_breakdown
        assert isinstance(logging_obj.cost_breakdown["additional_costs"], dict)
        
        # Check that the Azure Model Router flat cost is in additional_costs
        additional_costs = logging_obj.cost_breakdown["additional_costs"]
        assert "Azure Model Router Flat Cost" in additional_costs
        
        # Verify the flat cost value
        expected_flat_cost = (
            5000 * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )
        actual_flat_cost = additional_costs["Azure Model Router Flat Cost"]
        assert actual_flat_cost == pytest.approx(expected_flat_cost, rel=1e-9)
        
        print(f"Additional costs in breakdown: {additional_costs}")
        print(f"Azure Model Router Flat Cost: ${actual_flat_cost:.6f}")
