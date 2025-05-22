# What is this?
## Helper utilities for cost_per_token()

from typing import Literal, Optional, Tuple, cast

import litellm
from litellm import verbose_logger
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def _is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
        return True
    return False


def _is_above_200k(tokens: float) -> bool:
    if tokens > 200000:
        return True
    return False


def select_cost_metric_for_model(
    model_info: ModelInfo,
) -> Literal["cost_per_character", "cost_per_token"]:
    """
    Select 'cost_per_character' if model_info has 'input_cost_per_character'
    Select 'cost_per_token' if model_info has 'input_cost_per_token'
    """
    if model_info.get("input_cost_per_character"):
        return "cost_per_character"
    elif model_info.get("input_cost_per_token"):
        return "cost_per_token"
    else:
        raise ValueError(
            f"Model {model_info['key']} does not have 'input_cost_per_character' or 'input_cost_per_token'"
        )


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


def _check_tokens_above_threshold(
        model_info: ModelInfo,
        prefix: str,
        token_count: int,
        default_cost: float
) -> float:
    """
    Check if the number of tokens is above a threshold specified in model_info.
    Return the appropriate cost per token based on thresholds.

    Args:
        model_info: Dictionary containing model pricing information
        prefix: Prefix for the cost keys to check (e.g., "input_cost_per_token_above_")
        token_count: Number of tokens to check against thresholds
        default_cost: Default cost to use if no threshold is exceeded

    Returns:
        The appropriate cost per token based on whether thresholds are exceeded
    """
    for key, value in sorted(model_info.items(), reverse=True):
        if key.startswith(prefix) and value is not None:
            try:
                # Handle both formats: _above_128k_tokens and _above_128_tokens
                threshold_str = key.split("_above_")[1].split("_tokens")[0]
                threshold = float(threshold_str.replace("k", "")) * (
                    1000 if "k" in threshold_str else 1
                )
                if token_count > threshold:
                    return cast(float, model_info.get(key, default_cost))
            except (IndexError, ValueError):
                continue
            except Exception:
                continue

    return default_cost


def _get_token_base_cost(model_info: ModelInfo, usage: Usage) -> Tuple[float, float]:
    """
    Return prompt and completion costs for a given model and usage.

    If input_tokens > threshold and `input_cost_per_token_above_[x]k_tokens` or `input_cost_per_token_above_[x]_tokens` is set,
    then we use the corresponding threshold cost.

    Similarly, if completion_tokens > threshold, we use the corresponding threshold cost for completion.
    """
    prompt_base_cost = model_info["input_cost_per_token"]
    completion_base_cost = model_info["output_cost_per_token"]

    ## CHECK IF PROMPT TOKENS ABOVE THRESHOLD
    prompt_base_cost = _check_tokens_above_threshold(
        model_info=model_info,
        prefix="input_cost_per_token_above_",
        token_count=usage.prompt_tokens,
        default_cost=prompt_base_cost
    )

    ## CHECK IF COMPLETION TOKENS ABOVE THRESHOLD
    completion_base_cost = _check_tokens_above_threshold(
        model_info=model_info,
        prefix="output_cost_per_token_above_",
        token_count=usage.completion_tokens,
        default_cost=completion_base_cost
    )

    return prompt_base_cost, completion_base_cost


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


def _get_cache_cost(
    model_info: ModelInfo, 
    cost_key_prefix: str, 
    token_count: Optional[float],
) -> float:
    """
    Get the appropriate cache cost based on token count thresholds.

    Args:
        model_info: Dictionary containing model pricing information
        cost_key_prefix: Prefix for the cache cost keys (e.g., "cache_read_input_token_cost")
        token_count: Number of tokens to check against thresholds

    Returns:
        The appropriate cost per token based on whether thresholds are exceeded
    """
    if token_count is None or token_count <= 0:
        return 0.0

    # Get the base cost
    base_cost_key = cost_key_prefix
    base_cost = model_info.get(base_cost_key, 0.0)

    # If base_cost is None, default to 0.0
    if base_cost is None:
        verbose_logger.debug(
            f"Cache cost key '{base_cost_key}' is None for model. Defaulting to 0.0"
        )
        base_cost = 0.0

    # Check for threshold-based costs (e.g., above_200k_tokens, above_128k_tokens)
    above_200k_key = f"{cost_key_prefix}_above_200k_tokens"
    above_128k_key = f"{cost_key_prefix}_above_128k_tokens"

    # Use token_count to determine pricing tier (tokens in cache, not in prompt)
    # This is because pricing is based on the number of tokens in the cache, not the total prompt tokens
    tokens_to_check = token_count

    # Special handling for audio caching costs
    is_audio_cache = "audio" in cost_key_prefix

    # For audio caching costs, if the threshold keys don't exist but we have the base audio cost,
    # we need to calculate the threshold costs based on the regular token threshold multipliers
    if is_audio_cache:
        # Check if we need to calculate threshold costs for audio
        if _is_above_200k(tokens=tokens_to_check) and above_200k_key not in model_info:
            # For audio caching costs above 200k tokens, we need to check if we have the regular token threshold costs
            regular_cost_key_prefix = cost_key_prefix.replace("_audio", "")
            regular_above_200k_key = f"{regular_cost_key_prefix}_above_200k_tokens"

            if regular_above_200k_key in model_info and base_cost > 0:
                # Calculate the multiplier from regular token costs
                regular_base_cost = model_info.get(regular_cost_key_prefix, 0.0)
                if regular_base_cost > 0:
                    multiplier = model_info[regular_above_200k_key] / regular_base_cost
                    # Apply the same multiplier to audio cache cost
                    return float(token_count) * (base_cost * multiplier)

        elif _is_above_128k(tokens=tokens_to_check) and above_128k_key not in model_info:
            # For audio caching costs above 128k tokens, we need to check if we have the regular token threshold costs
            regular_cost_key_prefix = cost_key_prefix.replace("_audio", "")
            regular_above_128k_key = f"{regular_cost_key_prefix}_above_128k_tokens"

            if regular_above_128k_key in model_info and base_cost > 0:
                # Calculate the multiplier from regular token costs
                regular_base_cost = model_info.get(regular_cost_key_prefix, 0.0)
                if regular_base_cost > 0:
                    multiplier = model_info[regular_above_128k_key] / regular_base_cost
                    # Apply the same multiplier to audio cache cost
                    return float(token_count) * (base_cost * multiplier)

    # If we have input_cost_per_token_above_200k_tokens but no cache_read_input_token_cost_above_200k_tokens,
    # calculate the cache cost based on the ratio between base cache cost and base input cost
    if _is_above_200k(tokens=tokens_to_check) and above_200k_key not in model_info:
        input_cost_key = "input_cost_per_token"
        input_cost_above_200k_key = "input_cost_per_token_above_200k_tokens"

        if input_cost_above_200k_key in model_info and input_cost_key in model_info:
            base_input_cost = model_info.get(input_cost_key)
            above_200k_input_cost = model_info.get(input_cost_above_200k_key)

            if (base_input_cost is not None and above_200k_input_cost is not None and 
                base_input_cost > 0 and base_cost > 0):
                # Calculate the ratio between cache cost and input cost
                ratio = base_cost / base_input_cost
                # Apply the same ratio to the above_200k input cost
                return float(token_count) * (above_200k_input_cost * ratio)

    # Similarly for 128k threshold
    if _is_above_128k(tokens=tokens_to_check) and above_128k_key not in model_info:
        input_cost_key = "input_cost_per_token"
        input_cost_above_128k_key = "input_cost_per_token_above_128k_tokens"

        if input_cost_above_128k_key in model_info and input_cost_key in model_info:
            base_input_cost = model_info.get(input_cost_key)
            above_128k_input_cost = model_info.get(input_cost_above_128k_key)

            if (base_input_cost is not None and above_128k_input_cost is not None and 
                base_input_cost > 0 and base_cost > 0):
                # Calculate the ratio between cache cost and input cost
                ratio = base_cost / base_input_cost
                # Apply the same ratio to the above_128k input cost
                return float(token_count) * (above_128k_input_cost * ratio)

    # Standard threshold-based pricing logic
    if _is_above_200k(tokens=tokens_to_check) and above_200k_key in model_info:
        cost_value = model_info.get(above_200k_key)
        if cost_value is not None:
            return float(token_count) * cost_value
        return 0.0
    elif _is_above_128k(tokens=tokens_to_check) and above_128k_key in model_info:
        cost_value = model_info.get(above_128k_key)
        if cost_value is not None:
            return float(token_count) * cost_value
        return 0.0
    else:
        return float(token_count) * base_cost

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

    prompt_base_cost, completion_base_cost = _get_token_base_cost(
        model_info=model_info, usage=usage
    )

    prompt_cost = float(text_tokens) * prompt_base_cost

    ### CACHE READ COST
    # Handle regular text tokens cache read cost
    if cache_hit_tokens > 0:
        prompt_cost += _get_cache_cost(
            model_info=model_info,
            cost_key_prefix="cache_read_input_token_cost",
            token_count=cache_hit_tokens,
        )

    # Handle audio tokens cache read cost if applicable
    # Only calculate audio cache cost if audio tokens are actually cached
    # This is determined by checking if there are cached_audio_tokens in prompt_tokens_details
    cached_audio_tokens = 0
    if usage.prompt_tokens_details and hasattr(usage.prompt_tokens_details, "cached_audio_tokens"):
        cached_audio_tokens = getattr(usage.prompt_tokens_details, "cached_audio_tokens") or 0

    if cached_audio_tokens > 0:
        # Check if we have a specific audio cache cost
        if "cache_read_input_audio_token_cost" in model_info:
            prompt_cost += _get_cache_cost(
                model_info=model_info,
                cost_key_prefix="cache_read_input_audio_token_cost",
                token_count=cached_audio_tokens,
            )
        # If no specific audio cache cost, but we have a regular cache cost and audio input cost,
        # calculate the audio cache cost based on the ratio between regular cache cost and regular input cost
        elif "cache_read_input_token_cost" in model_info and "input_cost_per_audio_token" in model_info and "input_cost_per_token" in model_info:
            # Get the ratio between audio input cost and regular input cost
            audio_to_regular_ratio = model_info["input_cost_per_audio_token"] / model_info["input_cost_per_token"]
            # Apply this ratio to the cache cost
            audio_cache_cost = model_info["cache_read_input_token_cost"] * audio_to_regular_ratio
            # Create a temporary model_info with the calculated audio cache cost
            temp_model_info = model_info.copy()
            temp_model_info["cache_read_input_audio_token_cost"] = audio_cache_cost
            # Calculate the cost using the temporary model_info
            prompt_cost += _get_cache_cost(
                model_info=temp_model_info,
                cost_key_prefix="cache_read_input_audio_token_cost",
                token_count=cached_audio_tokens,
            )

    ### AUDIO COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_audio_token", audio_tokens
    )

    ### CACHE WRITING COST
    # Handle regular text tokens cache creation cost
    if usage._cache_creation_input_tokens and usage._cache_creation_input_tokens > 0:
        prompt_cost += _get_cache_cost(
            model_info=model_info,
            cost_key_prefix="cache_creation_input_token_cost",
            token_count=usage._cache_creation_input_tokens,
        )

    # Handle audio tokens cache creation cost if applicable
    # Only calculate audio cache creation cost if audio tokens are actually being cached
    # This is determined by checking if there are cached_audio_tokens in _cache_creation_audio_tokens
    cached_creation_audio_tokens = 0
    if hasattr(usage, "_cache_creation_audio_tokens"):
        cached_creation_audio_tokens = getattr(usage, "_cache_creation_audio_tokens") or 0

    if cached_creation_audio_tokens > 0:
        # Check if we have a specific audio cache creation cost
        if "cache_creation_input_audio_token_cost" in model_info:
            prompt_cost += _get_cache_cost(
                model_info=model_info,
                cost_key_prefix="cache_creation_input_audio_token_cost",
                token_count=cached_creation_audio_tokens,
            )
        # If no specific audio cache creation cost, but we have a regular cache creation cost and audio input cost,
        # calculate the audio cache creation cost based on the ratio between regular cache cost and regular input cost
        elif "cache_creation_input_token_cost" in model_info and "input_cost_per_audio_token" in model_info and "input_cost_per_token" in model_info:
            # Get the ratio between audio input cost and regular input cost
            audio_to_regular_ratio = model_info["input_cost_per_audio_token"] / model_info["input_cost_per_token"]
            # Apply this ratio to the cache creation cost
            audio_cache_creation_cost = model_info["cache_creation_input_token_cost"] * audio_to_regular_ratio
            # Create a temporary model_info with the calculated audio cache creation cost
            temp_model_info = model_info.copy()
            temp_model_info["cache_creation_input_audio_token_cost"] = audio_cache_creation_cost
            # Calculate the cost using the temporary model_info
            prompt_cost += _get_cache_cost(
                model_info=temp_model_info,
                cost_key_prefix="cache_creation_input_audio_token_cost",
                token_count=cached_creation_audio_tokens,
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
    text_tokens = 0
    audio_tokens = 0
    reasoning_tokens = 0
    is_text_tokens_total = False
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
            or 0  # default to completion tokens, if this field is not set
        )
        reasoning_tokens = (
            cast(
                Optional[int],
                getattr(usage.completion_tokens_details, "reasoning_tokens", 0),
            )
            or 0
        )

    if text_tokens == 0:
        text_tokens = usage.completion_tokens
    if text_tokens == usage.completion_tokens:
        is_text_tokens_total = True
    ## TEXT COST
    completion_cost = float(text_tokens) * completion_base_cost

    _output_cost_per_audio_token: Optional[float] = model_info.get(
        "output_cost_per_audio_token"
    )

    _output_cost_per_reasoning_token: Optional[float] = model_info.get(
        "output_cost_per_reasoning_token"
    )

    ## AUDIO COST
    if not is_text_tokens_total and audio_tokens is not None and audio_tokens > 0:
        _output_cost_per_audio_token = (
            _output_cost_per_audio_token
            if _output_cost_per_audio_token is not None
            else completion_base_cost
        )
        completion_cost += float(audio_tokens) * _output_cost_per_audio_token

    ## REASONING COST
    if not is_text_tokens_total and reasoning_tokens and reasoning_tokens > 0:
        _output_cost_per_reasoning_token = (
            _output_cost_per_reasoning_token
            if _output_cost_per_reasoning_token is not None
            else completion_base_cost
        )
        completion_cost += float(reasoning_tokens) * _output_cost_per_reasoning_token

    return prompt_cost, completion_cost
