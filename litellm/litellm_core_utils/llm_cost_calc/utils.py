# What is this?
## Helper utilities for cost_per_token()

from typing import Optional, Tuple

import litellm
from litellm import verbose_logger
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def _is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
        return True
    return False


def _generic_cost_per_character(
    model: str,
    custom_llm_provider: str,
    prompt_characters: float,
    completion_characters: float,
    custom_prompt_cost: Optional[float],
    custom_completion_cost: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates cost per character for aspeech/speech calls.

    Calculates the cost per character for a given model, input messages, and response object.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, "vertex_ai-*"
        - prompt_characters: float, the number of input characters
        - completion_characters: float, the number of output characters

    Returns:
        Tuple[Optional[float], Optional[float]] - prompt_cost_in_usd, completion_cost_in_usd.
        - returns None if not able to calculate cost.

    Raises:
        Exception if 'input_cost_per_character' or 'output_cost_per_character' is missing from model_info
    """
    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## CALCULATE INPUT COST
    try:
        if custom_prompt_cost is None:
            assert (
                "input_cost_per_character" in model_info
                and model_info["input_cost_per_character"] is not None
            ), "model info for model={} does not have 'input_cost_per_character'-pricing\nmodel_info={}".format(
                model, model_info
            )
            custom_prompt_cost = model_info["input_cost_per_character"]

        prompt_cost = prompt_characters * custom_prompt_cost
    except Exception as e:
        verbose_logger.exception(
            "litellm.litellm_core_utils.llm_cost_calc.utils.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                str(e)
            )
        )

        prompt_cost = None

    ## CALCULATE OUTPUT COST
    try:
        if custom_completion_cost is None:
            assert (
                "output_cost_per_character" in model_info
                and model_info["output_cost_per_character"] is not None
            ), "model info for model={} does not have 'output_cost_per_character'-pricing\nmodel_info={}".format(
                model, model_info
            )
            custom_completion_cost = model_info["output_cost_per_character"]
        completion_cost = completion_characters * custom_completion_cost
    except Exception as e:
        verbose_logger.exception(
            "litellm.litellm_core_utils.llm_cost_calc.utils.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                str(e)
            )
        )

        completion_cost = None

    return prompt_cost, completion_cost


def _get_prompt_token_base_cost(model_info: ModelInfo, usage: Usage) -> float:
    """
    Return prompt cost for a given model and usage.

    If input_tokens > 128k and `input_cost_per_token_above_128k_tokens` is set, then we use the `input_cost_per_token_above_128k_tokens` field.
    """
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    if _is_above_128k(usage.prompt_tokens) and input_cost_per_token_above_128k_tokens:
        return input_cost_per_token_above_128k_tokens
    return model_info["input_cost_per_token"]


def _get_completion_token_base_cost(model_info: ModelInfo, usage: Usage) -> float:
    """
    Return prompt cost for a given model and usage.

    If input_tokens > 128k and `input_cost_per_token_above_128k_tokens` is set, then we use the `input_cost_per_token_above_128k_tokens` field.
    """
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        _is_above_128k(usage.completion_tokens)
        and output_cost_per_token_above_128k_tokens
    ):
        return output_cost_per_token_above_128k_tokens
    return model_info["output_cost_per_token"]


def generic_cost_per_token(
    model: str, usage: Usage, custom_llm_provider: str
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Handles context caching as well.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## CALCULATE INPUT COST
    ### Cost of processing (non-cache hit + cache hit) + Cost of cache-writing (cache writing)
    prompt_cost = 0.0
    ### PROCESSING COST
    non_cache_hit_tokens = usage.prompt_tokens
    cache_hit_tokens = 0
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        cache_hit_tokens = usage.prompt_tokens_details.cached_tokens
        non_cache_hit_tokens = non_cache_hit_tokens - cache_hit_tokens

    prompt_base_cost = _get_prompt_token_base_cost(model_info=model_info, usage=usage)

    prompt_cost = float(non_cache_hit_tokens) * prompt_base_cost

    _cache_read_input_token_cost = model_info.get("cache_read_input_token_cost")
    if (
        _cache_read_input_token_cost is not None
        and usage.prompt_tokens_details
        and usage.prompt_tokens_details.cached_tokens
    ):
        prompt_cost += (
            float(usage.prompt_tokens_details.cached_tokens)
            * _cache_read_input_token_cost
        )

    ### CACHE WRITING COST
    _cache_creation_input_token_cost = model_info.get("cache_creation_input_token_cost")
    if _cache_creation_input_token_cost is not None:
        prompt_cost += (
            float(usage._cache_creation_input_tokens) * _cache_creation_input_token_cost
        )

    ## CALCULATE OUTPUT COST
    completion_base_cost = _get_completion_token_base_cost(
        model_info=model_info, usage=usage
    )
    completion_cost = usage["completion_tokens"] * completion_base_cost

    return prompt_cost, completion_cost
