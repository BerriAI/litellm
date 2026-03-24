"""
Helper util for handling perplexity-specific cost calculation
- e.g.: citation tokens, search queries
"""

from typing import Tuple, Union

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    InputCostBreakdown,
    OutputCostBreakdown,
)
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def _safe_float_cast(
    value: Union[str, int, float, None, object], default: float = 0.0
) -> float:
    """Safely cast a value to float with proper type handling for mypy."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore
    except (ValueError, TypeError):
        return default


def cost_per_token(
    model: str, usage: Usage
) -> Tuple[InputCostBreakdown, OutputCostBreakdown]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing perplexity-specific usage information

    Returns:
        Tuple[InputCostBreakdown, OutputCostBreakdown] - granular input and output cost breakdowns
    """
    cost_info = getattr(usage, "cost", None)
    if cost_info is not None and isinstance(cost_info, dict):
        total_cost = cost_info.get("total_cost")
        if total_cost is not None:
            cost_val = float(total_cost)
            return (
                InputCostBreakdown(total=0.0),
                OutputCostBreakdown(total=cost_val, text_cost=cost_val),
            )

    model_info = get_model_info(model=model, custom_llm_provider="perplexity")

    ## CALCULATE INPUT COST
    input_cost_per_token = _safe_float_cast(model_info.get("input_cost_per_token"))
    text_cost: float = (usage.prompt_tokens or 0) * input_cost_per_token

    ## CITATION TOKENS COST
    citation_tokens = getattr(usage, "citation_tokens", 0) or 0
    citation_cost_value = model_info.get("citation_cost_per_token")
    citation_cost = 0.0
    if citation_tokens > 0 and citation_cost_value is not None:
        citation_cost = citation_tokens * _safe_float_cast(citation_cost_value)
        text_cost += citation_cost

    prompt_cost = text_cost

    ## CALCULATE OUTPUT COST
    output_cost_per_token = _safe_float_cast(model_info.get("output_cost_per_token"))
    output_text_cost: float = (usage.completion_tokens or 0) * output_cost_per_token

    ## REASONING TOKENS COST
    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
    if (
        reasoning_tokens == 0
        and hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
    ):
        reasoning_tokens = (
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        )

    reasoning_cost = 0.0
    reasoning_cost_value = model_info.get("output_cost_per_reasoning_token")
    if reasoning_tokens > 0 and reasoning_cost_value is not None:
        reasoning_cost = reasoning_tokens * _safe_float_cast(reasoning_cost_value)

    ## SEARCH QUERIES COST
    num_search_queries = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        num_search_queries = (
            getattr(usage.prompt_tokens_details, "web_search_requests", 0) or 0
        )

    search_cost = 0.0
    search_cost_value = model_info.get(
        "search_queries_cost_per_query"
    ) or model_info.get("search_context_cost_per_query")
    if num_search_queries > 0 and search_cost_value is not None:
        if isinstance(search_cost_value, dict):
            search_cost_per_query = (
                _safe_float_cast(search_cost_value.get("search_context_size_low", 0))
                / 1000
            )
        else:
            search_cost_per_query = _safe_float_cast(search_cost_value)
        search_cost = num_search_queries * search_cost_per_query

    completion_cost = output_text_cost + reasoning_cost + search_cost

    return (
        InputCostBreakdown(total=prompt_cost, text_cost=text_cost),
        OutputCostBreakdown(
            total=completion_cost,
            text_cost=output_text_cost + search_cost,
            reasoning_cost=reasoning_cost,
        ),
    )
