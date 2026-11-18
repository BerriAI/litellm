"""
Helper util for handling perplexity-specific cost calculation
- e.g.: citation tokens, search queries
"""

from datetime import datetime
from typing import Final

from litellm.litellm_core_utils.core_helpers import finite_reported_cost
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import BranchFacts, ReportedCost, branch_cost
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def stated_cost(usage: Usage) -> ReportedCost | None:
    """The cost Perplexity reports on ``usage.cost``, a number or a breakdown under ``total_cost``.
    ``ResponseAPIUsage.parse_cost`` may already have flattened the breakdown to a float, so both shapes count."""
    cost_info: Final[object] = getattr(usage, "cost", None)
    total: Final[object] = (
        cost_info.get("total_cost")
        if isinstance(cost_info, dict)
        else cost_info
        if isinstance(cost_info, (int, float)) and not isinstance(cost_info, bool)
        else None
    )
    reported: Final = None if total is None else finite_reported_cost(total)
    return None if reported is None else ReportedCost(source="usage.cost", usd=reported)


def pricing_row(model: str, usage: Usage) -> ModelInfo | None:
    """The row Perplexity's own math reads, ``None`` when it bills the reported cost and reads no row."""
    if stated_cost(usage) is not None:
        return None
    return get_model_info(model=model, custom_llm_provider="perplexity")


def cost_per_token(model: str, usage: Usage, current_time: datetime | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens: the cost Perplexity
    reported when it did, else citation tokens, reasoning tokens and search queries on top of the token rates, at the
    model's off_peak_pricing rates while one of its windows is open.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing perplexity-specific usage information
        - current_time: the moment the request is billed at; defaults to now, UTC

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(
        model,
        "perplexity",
        usage,
        pricing_row(model, usage),
        branch=BranchFacts(reported=stated_cost(usage)),
        current_time=current_time,
    )
