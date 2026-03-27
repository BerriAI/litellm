"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    InputCostBreakdown,
    OutputCostBreakdown,
    generic_cost_per_token,
)

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage
import litellm


def cost_per_token(
    model: str, usage: "Usage"
) -> Tuple[InputCostBreakdown, OutputCostBreakdown]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[InputCostBreakdown, OutputCostBreakdown] - granular input and output cost breakdowns
    """
    input_bd, output_bd = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="anthropic"
    )

    try:
        model_info = litellm.get_model_info(
            model=model, custom_llm_provider="anthropic"
        )
        provider_specific_entry: dict = model_info.get("provider_specific_entry") or {}

        multiplier = 1.0
        inference_geo = getattr(usage, "inference_geo", None)
        if (
            inference_geo
            and inference_geo.lower() not in ["global", "not_available"]
        ):
            multiplier *= provider_specific_entry.get(inference_geo.lower(), 1.0)
        speed = getattr(usage, "speed", None)
        if speed == "fast":
            multiplier *= provider_specific_entry.get("fast", 1.0)

        if multiplier != 1.0:
            # Multiply text/audio/image input costs; leave cache costs unchanged
            input_bd["text_cost"] = input_bd.get("text_cost", 0.0) * multiplier
            input_bd["audio_cost"] = input_bd.get("audio_cost", 0.0) * multiplier
            input_bd["image_cost"] = input_bd.get("image_cost", 0.0) * multiplier
            input_bd["total"] = (
                input_bd.get("text_cost", 0.0)
                + input_bd.get("cache_read_cost", 0.0)
                + input_bd.get("cache_creation_cost", 0.0)
                + input_bd.get("audio_cost", 0.0)
                + input_bd.get("image_cost", 0.0)
            )

            # Multiply all output cost components
            output_bd["text_cost"] = output_bd.get("text_cost", 0.0) * multiplier
            output_bd["reasoning_cost"] = (
                output_bd.get("reasoning_cost", 0.0) * multiplier
            )
            output_bd["audio_cost"] = output_bd.get("audio_cost", 0.0) * multiplier
            output_bd["image_cost"] = output_bd.get("image_cost", 0.0) * multiplier
            output_bd["total"] = (
                output_bd.get("text_cost", 0.0)
                + output_bd.get("reasoning_cost", 0.0)
                + output_bd.get("audio_cost", 0.0)
                + output_bd.get("image_cost", 0.0)
            )
    except Exception:
        pass

    return input_bd, output_bd


def get_cost_for_anthropic_web_search(
    model_info: Optional["ModelInfo"] = None,
    usage: Optional["Usage"] = None,
) -> float:
    """
    Get the cost of using a web search tool for Anthropic.
    """
    from litellm.types.utils import SearchContextCostPerQuery

    ## Check if web search requests are in the usage object
    if model_info is None:
        return 0.0

    if (
        usage is None
        or usage.server_tool_use is None
        or usage.server_tool_use.web_search_requests is None
    ):
        return 0.0

    ## Get the cost per web search request
    search_context_pricing: SearchContextCostPerQuery = (
        model_info.get("search_context_cost_per_query") or SearchContextCostPerQuery()
    )
    cost_per_web_search_request = search_context_pricing.get(
        "search_context_size_medium", 0.0
    )
    if cost_per_web_search_request is None or cost_per_web_search_request == 0.0:
        return 0.0

    ## Calculate the total cost
    total_cost = cost_per_web_search_request * usage.server_tool_use.web_search_requests
    return total_cost
