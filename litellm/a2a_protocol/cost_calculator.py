"""
Cost calculator for A2A (Agent-to-Agent) calls.
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

        Default is 0.0. In the future, users can configure cost per agent call.
        """
        if litellm_logging_obj is None:
            return 0.0

        # Check if user set a custom response cost
        response_cost = litellm_logging_obj.model_call_details.get(
            "response_cost", None
        )
        if response_cost is not None:
            return response_cost

        # Default to 0.0 for A2A calls
        return 0.0
