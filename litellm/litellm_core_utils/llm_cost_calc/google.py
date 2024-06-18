# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
from typing import Literal, Tuple

import litellm

"""
Gemini pricing covers: 
- token
- image
- audio
- video
"""

models_without_dynamic_pricing = ["gemini-1.0-pro", "gemini-pro"]


def _is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
        return True
    return False


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    prompt_tokens: float,
    completion_tokens: float,
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """
    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## CALCULATE INPUT COST
    if (
        _is_above_128k(tokens=prompt_tokens)
        and model not in models_without_dynamic_pricing
    ):
        assert (
            model_info["input_cost_per_token_above_128k_tokens"] is not None
        ), "model info for model={} does not have pricing for > 128k tokens\nmodel_info={}".format(
            model, model_info
        )
        prompt_cost = (
            prompt_tokens * model_info["input_cost_per_token_above_128k_tokens"]
        )
    else:
        prompt_cost = prompt_tokens * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    if (
        _is_above_128k(tokens=completion_tokens)
        and model not in models_without_dynamic_pricing
    ):
        assert (
            model_info["output_cost_per_token_above_128k_tokens"] is not None
        ), "model info for model={} does not have pricing for > 128k tokens\nmodel_info={}".format(
            model, model_info
        )
        completion_cost = (
            completion_tokens * model_info["output_cost_per_token_above_128k_tokens"]
        )
    else:
        completion_cost = completion_tokens * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
