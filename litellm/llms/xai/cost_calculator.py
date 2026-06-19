"""
Helper util for handling XAI-specific cost calculation
- Uses the generic cost calculator which already handles tiered pricing correctly
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

from typing import TYPE_CHECKING, Mapping, Tuple

from litellm.types.utils import Usage
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo

# https://docs.x.ai/developers/pricing#tools-pricing
_WEB_SEARCH_COST_PER_CALL = 5.0 / 1000.0


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.
    Uses the generic cost calculator for all pricing logic, with XAI-specific reasoning token handling.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    # XAI-specific completion cost: completion is billed as visible + reasoning
    # tokens. Detect when the transformation layer already folded them so we
    # don't double-count; fall back to raw xAI shape for callers that bypass
    # the transformation (e.g. proxy logs replayed into cost calc).
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        )

    already_normalised = total_tokens == prompt_tokens + completion_tokens
    total_completion_tokens = (
        completion_tokens
        if already_normalised
        else completion_tokens + reasoning_tokens
    )

    modified_usage = Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=usage.prompt_tokens_details,
        completion_tokens_details=None,
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=modified_usage, custom_llm_provider="xai"
    )

    return prompt_cost, completion_cost


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of web search requests for X.AI models.

    Uses usage.server_side_tool_usage_details.web_search_calls at $5 / 1k calls
    (xAI tools pricing), not legacy num_sources_used / web_search_requests.
    """
    _ = model_info
    details = getattr(usage, "server_side_tool_usage_details", None)
    if not isinstance(details, Mapping):
        return 0.0
    try:
        web_search_calls = int(details.get("web_search_calls") or 0)
    except (TypeError, ValueError):
        return 0.0
    if web_search_calls <= 0:
        return 0.0
    return _WEB_SEARCH_COST_PER_CALL * web_search_calls
