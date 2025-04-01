# What is this?
## Helper utilities for cost_per_token()

from typing import Optional, Tuple, cast

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


def calculate_cost_component(
    model_info: ModelInfo, cost_key: str, usage_value: Optional[float]
) -> float:
    """
    Generic cost calculator for any usage component

    Args:
        model_info: Dictionary containing cost information
        cost_key: The key for the cost multiplier in model_info (e.g., 'input_cost_per_audio_token')
        usage_value: The actual usage value (e.g., number of tokens, characters, seconds)

    Returns:
        float: The calculated cost
    """
    cost_per_unit = model_info.get(cost_key)
    if (
        cost_per_unit is not None
        and isinstance(cost_per_unit, float)
        and usage_value is not None
        and usage_value > 0
    ):
        return float(usage_value) * cost_per_unit
    return 0.0


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
    text_tokens = usage.prompt_tokens
    cache_hit_tokens = 0
    audio_tokens = 0
    character_count = 0
    image_count = 0
    video_length_seconds = 0
    if usage.prompt_tokens_details:
        cache_hit_tokens = (
            cast(
                Optional[int], getattr(usage.prompt_tokens_details, "cached_tokens", 0)
            )
            or 0
        )
        text_tokens = (
            cast(
                Optional[int], getattr(usage.prompt_tokens_details, "text_tokens", None)
            )
            or 0  # default to prompt tokens, if this field is not set
        )
        audio_tokens = (
            cast(Optional[int], getattr(usage.prompt_tokens_details, "audio_tokens", 0))
            or 0
        )
        character_count = (
            cast(
                Optional[int],
                getattr(usage.prompt_tokens_details, "character_count", 0),
            )
            or 0
        )
        image_count = (
            cast(Optional[int], getattr(usage.prompt_tokens_details, "image_count", 0))
            or 0
        )
        video_length_seconds = (
            cast(
                Optional[int],
                getattr(usage.prompt_tokens_details, "video_length_seconds", 0),
            )
            or 0
        )

    ## EDGE CASE - text tokens not set inside PromptTokensDetails
    if text_tokens == 0:
        text_tokens = usage.prompt_tokens - cache_hit_tokens - audio_tokens

    prompt_base_cost = _get_prompt_token_base_cost(model_info=model_info, usage=usage)

    prompt_cost = float(text_tokens) * prompt_base_cost

    ### CACHE READ COST
    prompt_cost += calculate_cost_component(
        model_info, "cache_read_input_token_cost", cache_hit_tokens
    )

    ### AUDIO COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_audio_token", audio_tokens
    )

    ### CACHE WRITING COST
    prompt_cost += calculate_cost_component(
        model_info,
        "cache_creation_input_token_cost",
        usage._cache_creation_input_tokens,
    )

    ### CHARACTER COST

    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_character", character_count
    )

    ### IMAGE COUNT COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_image", image_count
    )

    ### VIDEO LENGTH COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_video_per_second", video_length_seconds
    )

    ## CALCULATE OUTPUT COST
    completion_base_cost = _get_completion_token_base_cost(
        model_info=model_info, usage=usage
    )
    text_tokens = usage.completion_tokens
    audio_tokens = 0
    if usage.completion_tokens_details is not None:
        audio_tokens = (
            cast(
                Optional[int],
                getattr(usage.completion_tokens_details, "audio_tokens", 0),
            )
            or 0
        )
        text_tokens = (
            cast(
                Optional[int],
                getattr(usage.completion_tokens_details, "text_tokens", None),
            )
            or usage.completion_tokens  # default to completion tokens, if this field is not set
        )

    ## TEXT COST
    completion_cost = float(text_tokens) * completion_base_cost

    _output_cost_per_audio_token: Optional[float] = model_info.get(
        "output_cost_per_audio_token"
    )

    ## AUDIO COST
    if (
        _output_cost_per_audio_token is not None
        and audio_tokens is not None
        and audio_tokens > 0
    ):
        completion_cost += float(audio_tokens) * _output_cost_per_audio_token

    return prompt_cost, completion_cost
