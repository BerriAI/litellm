"""
Helper util for handling openai-specific cost calculation
- e.g.: prompt caching
"""

from typing import Literal

from litellm.litellm_core_utils.llm_cost_calc.utils import cost_per_text
from litellm.types.utils import CallTypes, Usage


def cost_router(call_type: CallTypes) -> Literal["cost_per_token", "cost_per_second"]:
    if call_type == CallTypes.atranscription or call_type == CallTypes.transcription:
        return "cost_per_second"
    else:
        return "cost_per_token"


def cost_per_token(
    model: str,
    usage: Usage,
    service_tier: str | None = None,
    data_residency: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - data_residency: optional OpenAI data-residency region (e.g. "eu", "us"),
          inferred from api_base. Applies the model's regional-processing
          uplift multiplier when set.

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return cost_per_text(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        service_tier=service_tier,
        data_residency=data_residency,
    )
