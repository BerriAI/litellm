"""
Helper util for handling XAI-specific cost calculation
- Uses the generic cost calculator which already handles tiered pricing correctly
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

from typing import TYPE_CHECKING, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    InputCostBreakdown,
    OutputCostBreakdown,
    generic_cost_per_token,
)
from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def cost_per_token(
    model: str, usage: Usage
) -> Tuple[InputCostBreakdown, OutputCostBreakdown]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.
    Uses the generic cost calculator for all pricing logic, with XAI-specific reasoning token handling.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[InputCostBreakdown, OutputCostBreakdown] - granular input and output cost breakdowns
    """
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        )

    total_completion_tokens = completion_tokens + reasoning_tokens

    modified_usage = Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=usage.prompt_tokens_details,
        completion_tokens_details=None,
    )

    return generic_cost_per_token(
        model=model, usage=modified_usage, custom_llm_provider="xai"
    )


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of web search requests for X.AI models.

    X.AI Live Search costs $25 per 1,000 sources used.
    Each source costs $0.025.

    The number of sources is stored in prompt_tokens_details.web_search_requests
    by the transformation layer to be compatible with the existing detection system.
    """
    # Cost per source used: $25 per 1,000 sources = $0.025 per source
    cost_per_source = 25.0 / 1000.0  # $0.025

    num_sources_used = 0

    if (
        hasattr(usage, "prompt_tokens_details")
        and usage.prompt_tokens_details is not None
        and hasattr(usage.prompt_tokens_details, "web_search_requests")
        and usage.prompt_tokens_details.web_search_requests is not None
    ):
        num_sources_used = int(usage.prompt_tokens_details.web_search_requests)

    # Fallback: try to get from num_sources_used if set directly
    elif hasattr(usage, "num_sources_used") and getattr(usage, "num_sources_used", None) is not None:
        num_sources_used = int(getattr(usage, "num_sources_used"))

    total_cost = cost_per_source * num_sources_used

    return total_cost
