"""
Cost calculator for A2A (Agent-to-Agent) calls.

Supports dynamic cost parameters that allow platform owners
to define custom costs per agent query or per token.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, SupportsFloat, SupportsIndex

from pydantic import TypeAdapter
from typing_extensions import Buffer

from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


A2A_COST_PARAM_KEYS: Final = ("cost_per_query", "input_cost_per_token", "output_cost_per_token")
_PRICE: Final = TypeAdapter(float)


def a2a_cost_params(litellm_params: Mapping[str, object] | None) -> Mapping[str, object]:
    """Only the agent's pricing keys reach the logging object; its credentials never do."""
    return MappingProxyType(
        {
            key: litellm_params[key]
            for key in A2A_COST_PARAM_KEYS
            if litellm_params is not None and litellm_params.get(key) is not None
        }
    )


def _price(cost_params: Mapping[str, object], key: str) -> float | None:
    value: Final = cost_params.get(key)
    return None if value is None or value == "" else _PRICE.validate_python(value)


def _custom_response_cost(model_call_details: Mapping[str, object]) -> float | None:
    response_cost: Final = model_call_details.get("response_cost", None)
    if response_cost is None:
        return None
    if isinstance(response_cost, (str, Buffer, SupportsFloat, SupportsIndex)):
        return float(response_cost)
    raise TypeError(f"float() argument must be a string or a real number, not {type(response_cost).__name__!r}")


class A2ACostCalculator:
    @staticmethod
    def calculate_a2a_cost(
        litellm_logging_obj: LitellmLoggingObject | None,
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

        model_call_details: Final = litellm_logging_obj.model_call_details

        # Check if user set a custom response cost (backward compatibility)
        custom_response_cost: Final = _custom_response_cost(model_call_details)
        if custom_response_cost is not None:
            return custom_response_cost

        usage: Final = model_call_details.get("usage")
        return A2ACostCalculator.agent_cost(
            cost_params=model_call_details.get("litellm_params", {}) or {},
            usage=usage if isinstance(usage, Usage) else None,
        )

    @staticmethod
    def agent_cost(cost_params: Mapping[str, object], usage: Usage | None) -> float:
        cost_per_query: Final = _price(cost_params, "cost_per_query")
        if cost_per_query is not None:
            return cost_per_query
        input_cost_per_token: Final = _price(cost_params, "input_cost_per_token")
        output_cost_per_token: Final = _price(cost_params, "output_cost_per_token")
        if usage is None or (input_cost_per_token is None and output_cost_per_token is None):
            return 0.0
        return usage.prompt_tokens * (input_cost_per_token or 0.0) + usage.completion_tokens * (
            output_cost_per_token or 0.0
        )
