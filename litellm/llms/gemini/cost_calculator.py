"""
This file is used to calculate the cost of the Gemini API.

Handles the context caching for Gemini API.
"""

from typing import Tuple

import litellm
from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _is_above_128k,
    _is_above_200k,
    generic_cost_per_token,
)
from litellm.types.utils import ModelInfo, Usage


def _handle_200k_pricing(
    model_info: ModelInfo,
    usage: Usage,
) -> Tuple[float, float]:
    ## CALCULATE INPUT COST
    input_cost_per_token_above_200k_tokens = model_info.get(
        "input_cost_per_token_above_200k_tokens"
    )
    output_cost_per_token_above_200k_tokens = model_info.get(
        "output_cost_per_token_above_200k_tokens"
    )

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    if (
        _is_above_200k(tokens=prompt_tokens)
        and input_cost_per_token_above_200k_tokens is not None
    ):
        prompt_cost = prompt_tokens * input_cost_per_token_above_200k_tokens
    else:
        prompt_cost = prompt_tokens * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    output_cost_per_token_above_200k_tokens = model_info.get(
        "output_cost_per_token_above_200k_tokens"
    )
    if (
        _is_above_200k(tokens=completion_tokens)
        and output_cost_per_token_above_200k_tokens is not None
    ):
        completion_cost = completion_tokens * output_cost_per_token_above_200k_tokens
    else:
        completion_cost = completion_tokens * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost


def _handle_128k_pricing(
    model_info: ModelInfo,
    usage: Usage,
) -> Tuple[float, float]:
    ## CALCULATE INPUT COST
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    if (
        _is_above_128k(tokens=prompt_tokens)
        and input_cost_per_token_above_128k_tokens is not None
    ):
        prompt_cost = prompt_tokens * input_cost_per_token_above_128k_tokens
    else:
        prompt_cost = prompt_tokens * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        _is_above_128k(tokens=completion_tokens)
        and output_cost_per_token_above_128k_tokens is not None
    ):
        completion_cost = completion_tokens * output_cost_per_token_above_128k_tokens
    else:
        completion_cost = completion_tokens * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: Usage object containing token counts

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider="gemini"
    )

    ## HANDLE 200k+ PRICING FOR GEMINI-2.5-PRO MODELS
    input_cost_per_token_above_200k_tokens = model_info.get(
        "input_cost_per_token_above_200k_tokens"
    )
    output_cost_per_token_above_200k_tokens = model_info.get(
        "output_cost_per_token_above_200k_tokens"
    )
    if (
        input_cost_per_token_above_200k_tokens is not None
        or output_cost_per_token_above_200k_tokens is not None
    ):
        return _handle_200k_pricing(
            model_info=model_info,
            usage=usage,
        )

    ## HANDLE 128k+ PRICING
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        input_cost_per_token_above_128k_tokens is not None
        or output_cost_per_token_above_128k_tokens is not None
    ):
        return _handle_128k_pricing(
            model_info=model_info,
            usage=usage,
        )

    return generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="gemini"
    )
