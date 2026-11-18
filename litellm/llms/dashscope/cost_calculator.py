"""
Cost calculator for Dashscope Chat models.

Alibaba Model Studio tiered pricing is all-or-nothing: the tier is picked from the
total input tokens of a single request, and every token of that request (input,
cached, cache-creation, output, reasoning) is billed at that one tier's rate.
See https://help.aliyun.com/zh/model-studio/billing-for-model-studio
"""

from datetime import datetime

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str = "dashscope",
    current_time: datetime | None = None,
) -> tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.

    Supports both tiered and flat pricing with cached and reasoning tokens, and swaps in the
    model's off_peak_pricing rates while one of its windows is open.

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block
        custom_llm_provider: The provider id the request resolved to; dashscope or one of its brand aliases
        current_time: The moment the request is billed at; defaults to now, UTC

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    return branch_cost(
        model,
        custom_llm_provider,
        usage,
        get_model_info(model=model, custom_llm_provider=custom_llm_provider),
        current_time=current_time,
    )
