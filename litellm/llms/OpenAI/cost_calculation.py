"""
Helper util for handling openai-specific cost calculation
- e.g.: prompt caching
"""

from typing import Literal, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.types.utils import CallTypes, Usage
from litellm.utils import get_model_info


def cost_router(call_type: CallTypes) -> Literal["cost_per_token", "cost_per_second"]:
    if call_type == CallTypes.atranscription or call_type == CallTypes.transcription:
        return "cost_per_second"
    else:
        return "cost_per_token"


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="openai")

    ## CALCULATE INPUT COST
    ### Non-cached text tokens
    non_cached_text_tokens = usage.prompt_tokens
    cached_tokens: Optional[int] = None
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        cached_tokens = usage.prompt_tokens_details.cached_tokens
        non_cached_text_tokens = non_cached_text_tokens - cached_tokens
    prompt_cost: float = non_cached_text_tokens * model_info["input_cost_per_token"]
    ## Prompt Caching cost calculation
    if model_info.get("cache_read_input_token_cost") is not None and cached_tokens:
        # Note: We read ._cache_read_input_tokens from the Usage - since cost_calculator.py standardizes the cache read tokens on usage._cache_read_input_tokens
        prompt_cost += cached_tokens * (
            model_info.get("cache_read_input_token_cost", 0) or 0
        )

    _audio_tokens: Optional[int] = (
        usage.prompt_tokens_details.audio_tokens
        if usage.prompt_tokens_details is not None
        else None
    )
    _audio_cost_per_token: Optional[float] = model_info.get(
        "input_cost_per_audio_token"
    )
    if _audio_tokens is not None and _audio_cost_per_token is not None:
        audio_cost: float = _audio_tokens * _audio_cost_per_token
        prompt_cost += audio_cost

    ## CALCULATE OUTPUT COST
    completion_cost: float = (
        usage["completion_tokens"] * model_info["output_cost_per_token"]
    )
    _output_cost_per_audio_token: Optional[float] = model_info.get(
        "output_cost_per_audio_token"
    )
    _output_audio_tokens: Optional[int] = (
        usage.completion_tokens_details.audio_tokens
        if usage.completion_tokens_details is not None
        else None
    )
    if _output_cost_per_audio_token is not None and _output_audio_tokens is not None:
        audio_cost = _output_audio_tokens * _output_cost_per_audio_token
        completion_cost += audio_cost

    return prompt_cost, completion_cost


def cost_per_second(
    model: str, usage: Usage, response_time_ms: Optional[float] = 0.0
) -> Tuple[float, float]:
    """
    Calculates the cost per second for a given model, prompt tokens, and completion tokens.
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="openai")
    prompt_cost = 0.0
    completion_cost = 0.0
    ## Speech / Audio cost calculation
    if (
        "output_cost_per_second" in model_info
        and model_info["output_cost_per_second"] is not None
        and response_time_ms is not None
    ):
        verbose_logger.debug(
            f"For model={model} - output_cost_per_second: {model_info.get('output_cost_per_second')}; response time: {response_time_ms}"
        )
        ## COST PER SECOND ##
        completion_cost = model_info["output_cost_per_second"] * response_time_ms / 1000
    elif (
        "input_cost_per_second" in model_info
        and model_info["input_cost_per_second"] is not None
        and response_time_ms is not None
    ):
        verbose_logger.debug(
            f"For model={model} - input_cost_per_second: {model_info.get('input_cost_per_second')}; response time: {response_time_ms}"
        )
        ## COST PER SECOND ##
        prompt_cost = model_info["input_cost_per_second"] * response_time_ms / 1000
        completion_cost = 0.0

    return prompt_cost, completion_cost
