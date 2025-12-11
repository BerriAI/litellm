"""
Cost calculator for A2A (Agent-to-Agent) calls.

Supports dynamic cost parameters that allow platform owners
to define custom costs per agent query or per token.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


class A2ACostCalculator:
    @staticmethod
    def calculate_a2a_cost(
        litellm_logging_obj: Optional[LitellmLoggingObject],
    ) -> float:
        """
        Calculate the cost of an A2A send_message call.

        Supports multiple cost parameters for platform owners:
        - cost_per_query: Fixed cost per query
        - input_cost_per_token + output_cost_per_token: Token-based pricing

        Priority order:
        1. response_cost - if set directly (backward compatibility)
        2. cost_per_query - fixed cost per query
        3. input_cost_per_token + output_cost_per_token - token-based cost
        4. Default to 0.0

        Args:
            litellm_logging_obj: The LiteLLM logging object containing call details

        Returns:
            float: The cost of the A2A call
        """
        if litellm_logging_obj is None:
            return 0.0

        model_call_details = litellm_logging_obj.model_call_details

        # Check if user set a custom response cost (backward compatibility)
        response_cost = model_call_details.get("response_cost", None)
        if response_cost is not None:
            return float(response_cost)

        # Get litellm_params for cost parameters
        litellm_params = model_call_details.get("litellm_params", {}) or {}

        # Check for cost_per_query (fixed cost per query)
        if litellm_params.get("cost_per_query") is not None:
            return float(litellm_params["cost_per_query"])

        # Check for token-based pricing
        input_cost_per_token = litellm_params.get("input_cost_per_token")
        output_cost_per_token = litellm_params.get("output_cost_per_token")

        if input_cost_per_token is not None or output_cost_per_token is not None:
            return A2ACostCalculator._calculate_token_based_cost(
                model_call_details=model_call_details,
                input_cost_per_token=input_cost_per_token,
                output_cost_per_token=output_cost_per_token,
            )

        # Default to 0.0 for A2A calls
        return 0.0

    @staticmethod
    def _calculate_token_based_cost(
        model_call_details: dict,
        input_cost_per_token: Optional[float],
        output_cost_per_token: Optional[float],
    ) -> float:
        """
        Calculate cost based on token usage and per-token pricing.

        Args:
            model_call_details: The model call details containing usage
            input_cost_per_token: Cost per input token (can be None, defaults to 0)
            output_cost_per_token: Cost per output token (can be None, defaults to 0)

        Returns:
            float: The calculated cost
        """
        # Get usage from model_call_details
        usage = model_call_details.get("usage")
        if usage is None:
            return 0.0

        # Get token counts
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Calculate costs
        input_cost = prompt_tokens * (float(input_cost_per_token) if input_cost_per_token else 0.0)
        output_cost = completion_tokens * (float(output_cost_per_token) if output_cost_per_token else 0.0)

        return input_cost + output_cost
