"""
Cost calculator for A2A (Agent-to-Agent) calls.

Supports dynamic cost_per_query parameter that allows platform owners
to define custom costs per agent query.
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

        Supports cost_per_query parameter for platform owners to define
        custom costs per agent query.

        Priority order:
        1. response_cost - if set directly (backward compatibility)
        2. cost_per_query - in litellm_params
        3. Default to 0.0

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

        # Check for cost_per_query in litellm_params (standard location for cost params)
        litellm_params = model_call_details.get("litellm_params", {})
        if litellm_params and litellm_params.get("cost_per_query") is not None:
            return float(litellm_params["cost_per_query"])

        # Default to 0.0 for A2A calls
        return 0.0
