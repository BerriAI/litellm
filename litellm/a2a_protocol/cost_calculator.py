"""
Cost calculator for A2A (Agent-to-Agent) calls.

Supports dynamic cost parameters that allow platform owners
to define custom costs per agent query or per token.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, SupportsFloat, SupportsIndex

from pydantic import TypeAdapter
from typing_extensions import Buffer

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    A2aDocument,
    AgentUsage,
    price_fee,
    total_or_zero,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


def _custom_response_cost(model_call_details: Mapping[str, object]) -> float | None:
    response_cost: Final = model_call_details.get("response_cost", None)
    if response_cost is None:
        return None
    if isinstance(response_cost, (str, Buffer, SupportsFloat, SupportsIndex)):
        return float(response_cost)
    raise TypeError(f"float() argument must be a string or a real number, not {type(response_cost).__name__!r}")


_CONFIGURED_PRICE: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


def _configured_price(litellm_params: Mapping[str, object], key: str) -> float | None:
    return _CONFIGURED_PRICE.validate_python(litellm_params.get(key))


def _agent_usage(usage: object) -> AgentUsage | None:
    if usage is None:
        return None
    return AgentUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


class A2ACostCalculator:
    @staticmethod
    def calculate_a2a_cost(
        litellm_logging_obj: LitellmLoggingObject | None,
    ) -> float:
        """The cost of an A2A send_message call: a hook-set ``response_cost`` wins, then the agent's configured
        ``cost_per_query``, then its token prices over the call's usage, else 0.0."""
        if litellm_logging_obj is None:
            return 0.0

        model_call_details: Final = litellm_logging_obj.model_call_details
        custom_response_cost: Final = _custom_response_cost(model_call_details)
        if custom_response_cost is not None:
            return custom_response_cost

        litellm_params: Final[Mapping[str, object]] = model_call_details.get("litellm_params", {}) or {}
        return total_or_zero(
            price_fee(
                A2aDocument(
                    kind="a2a",
                    cost_per_query=_configured_price(litellm_params, "cost_per_query"),
                    input_cost_per_token=_configured_price(litellm_params, "input_cost_per_token") or None,
                    output_cost_per_token=_configured_price(litellm_params, "output_cost_per_token") or None,
                    usage=_agent_usage(model_call_details.get("usage")),
                )
            )
        )
