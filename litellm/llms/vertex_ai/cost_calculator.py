# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
from typing import Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_vertex_regional_endpoint_uplift,
    is_above_128k,
)
from litellm.types.utils import ModelInfo, Usage

"""
Gemini pricing covers: 
- token
- image
- audio
- video
"""


def _handle_128k_pricing(
    model_info: ModelInfo,
    usage: Usage,
) -> tuple[float, float]:
    ## CALCULATE INPUT COST
    input_cost_per_token_above_128k_tokens: Final = model_info.get("input_cost_per_token_above_128k_tokens")
    output_cost_per_token_above_128k_tokens = model_info.get("output_cost_per_token_above_128k_tokens")

    prompt_tokens: Final = usage.prompt_tokens
    completion_tokens: Final = usage.completion_tokens

    if is_above_128k(tokens=prompt_tokens) and input_cost_per_token_above_128k_tokens is not None:
        prompt_cost = prompt_tokens * input_cost_per_token_above_128k_tokens
    else:
        prompt_cost = prompt_tokens * (model_info["input_cost_per_token"] or 0.0)

    ## CALCULATE OUTPUT COST
    output_cost_per_token_above_128k_tokens = model_info.get("output_cost_per_token_above_128k_tokens")
    if is_above_128k(tokens=completion_tokens) and output_cost_per_token_above_128k_tokens is not None:
        completion_cost = completion_tokens * output_cost_per_token_above_128k_tokens
    else:
        completion_cost = completion_tokens * (model_info["output_cost_per_token"] or 0.0)

    return prompt_cost, completion_cost


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    service_tier: str | None = None,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """

    ## GET MODEL INFO
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## HANDLE 128k+ PRICING
    input_cost_per_token_above_128k_tokens: Final = model_info.get("input_cost_per_token_above_128k_tokens")
    output_cost_per_token_above_128k_tokens: Final = model_info.get("output_cost_per_token_above_128k_tokens")
    if input_cost_per_token_above_128k_tokens is not None or output_cost_per_token_above_128k_tokens is not None:
        prompt_cost_128k, completion_cost_128k = _handle_128k_pricing(
            model_info=model_info,
            usage=usage,
        )
        vertex_uplift: Final = get_vertex_regional_endpoint_uplift(model_info, vertex_location)
        return prompt_cost_128k * vertex_uplift, completion_cost_128k * vertex_uplift

    return generic_cost_per_token(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        service_tier=service_tier,
        vertex_location=vertex_location,
    )
