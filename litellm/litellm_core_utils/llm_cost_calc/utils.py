# What is this?
## Helper utilities for cost_per_token()

from typing import Literal, Optional, Tuple, TypedDict, cast

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CallTypes,
    ImageResponse,
    ModelInfo,
    PassthroughCallTypes,
    ServiceTier,
    Usage,
)
from litellm.utils import get_model_info


def _is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
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


def _get_service_tier_cost_key(base_key: str, service_tier: Optional[str]) -> str:
    """
    Get the appropriate cost key based on service tier.

    Args:
        base_key: The base cost key (e.g., "input_cost_per_token")
        service_tier: The service tier ("flex", "priority", or None for standard)

    Returns:
        str: The cost key to use (e.g., "input_cost_per_token_flex" or "input_cost_per_token")
    """
    if service_tier is None:
        return base_key

    # Only use service tier specific keys for "flex" and "priority"
    if service_tier.lower() in [ServiceTier.FLEX.value, ServiceTier.PRIORITY.value]:
        return f"{base_key}_{service_tier.lower()}"

    # For any other service tier, use standard pricing
    return base_key


def _get_token_base_cost(
    model_info: ModelInfo, usage: Usage, service_tier: Optional[str] = None
) -> Tuple[float, float, float, float, float]:
    """
    Return prompt cost, completion cost, and cache costs for a given model and usage.

    If input_tokens > threshold and `input_cost_per_token_above_[x]k_tokens` or `input_cost_per_token_above_[x]_tokens` is set,
    then we use the corresponding threshold cost for all token types.

    Returns:
        Tuple[float, float, float, float] - (prompt_cost, completion_cost, cache_creation_cost, cache_read_cost)
    """
    # Get service tier aware cost keys
    input_cost_key = _get_service_tier_cost_key("input_cost_per_token", service_tier)
    output_cost_key = _get_service_tier_cost_key("output_cost_per_token", service_tier)
    cache_creation_cost_key = _get_service_tier_cost_key(
        "cache_creation_input_token_cost", service_tier
    )
    cache_read_cost_key = _get_service_tier_cost_key(
        "cache_read_input_token_cost", service_tier
    )

    prompt_base_cost = cast(float, _get_cost_per_unit(model_info, input_cost_key))
    completion_base_cost = cast(float, _get_cost_per_unit(model_info, output_cost_key))
    cache_creation_cost = cast(
        float, _get_cost_per_unit(model_info, cache_creation_cost_key)
    )
    cache_creation_cost_above_1hr = cast(
        float,
        _get_cost_per_unit(model_info, "cache_creation_input_token_cost_above_1hr"),
    )
    cache_read_cost = cast(float, _get_cost_per_unit(model_info, cache_read_cost_key))

    ## CHECK IF ABOVE THRESHOLD
    threshold: Optional[float] = None
    for key, value in sorted(model_info.items(), reverse=True):
        if key.startswith("input_cost_per_token_above_") and value is not None:
            try:
                # Handle both formats: _above_128k_tokens and _above_128_tokens
                threshold_str = key.split("_above_")[1].split("_tokens")[0]
                threshold = float(threshold_str.replace("k", "")) * (
                    1000 if "k" in threshold_str else 1
                )
                if usage.prompt_tokens > threshold:

                    prompt_base_cost = cast(
                        float, _get_cost_per_unit(model_info, key, prompt_base_cost)
                    )
                    completion_base_cost = cast(
                        float,
                        _get_cost_per_unit(
                            model_info,
                            f"output_cost_per_token_above_{threshold_str}_tokens",
                            completion_base_cost,
                        ),
                    )

                    # Apply tiered pricing to cache costs
                    cache_creation_tiered_key = (
                        f"cache_creation_input_token_cost_above_{threshold_str}_tokens"
                    )
                    cache_read_tiered_key = (
                        f"cache_read_input_token_cost_above_{threshold_str}_tokens"
                    )

                    if cache_creation_tiered_key in model_info:
                        cache_creation_cost = cast(
                            float,
                            _get_cost_per_unit(
                                model_info,
                                cache_creation_tiered_key,
                                cache_creation_cost,
                            ),
                        )

                    if cache_read_tiered_key in model_info:
                        cache_read_cost = cast(
                            float,
                            _get_cost_per_unit(
                                model_info, cache_read_tiered_key, cache_read_cost
                            ),
                        )

                    break
            except (IndexError, ValueError):
                continue
            except Exception:
                continue

    return (
        prompt_base_cost,
        completion_base_cost,
        cache_creation_cost,
        cache_creation_cost_above_1hr,
        cache_read_cost,
    )


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
    cost_per_unit = _get_cost_per_unit(model_info, cost_key)
    if (
        cost_per_unit is not None
        and isinstance(cost_per_unit, float)
        and usage_value is not None
        and usage_value > 0
    ):
        return float(usage_value) * cost_per_unit
    return 0.0


def _get_cost_per_unit(
    model_info: ModelInfo, cost_key: str, default_value: Optional[float] = 0.0
) -> Optional[float]:
    # Sometimes the cost per unit is a string (e.g.: If a value like "3e-7" was read from the config.yaml)
    cost_per_unit = model_info.get(cost_key)
    if isinstance(cost_per_unit, float):
        return cost_per_unit
    if isinstance(cost_per_unit, int):
        return float(cost_per_unit)
    if isinstance(cost_per_unit, str):
        try:
            return float(cost_per_unit)
        except ValueError:
            verbose_logger.exception(
                f"litellm.litellm_core_utils.llm_cost_calc.utils.py::calculate_cost_per_component(): Exception occured - {cost_per_unit}\nDefaulting to 0.0"
            )

    # If the service tier key doesn't exist or is None, try to fall back to the standard key
    if cost_per_unit is None:
        # Check if any service tier suffix exists in the cost key using ServiceTier enum
        for service_tier in ServiceTier:
            suffix = f"_{service_tier.value}"
            if suffix in cost_key:
                # Extract the base key by removing the matched suffix
                base_key = cost_key.replace(suffix, "")
                fallback_cost = model_info.get(base_key)
                if isinstance(fallback_cost, float):
                    return fallback_cost
                if isinstance(fallback_cost, int):
                    return float(fallback_cost)
                if isinstance(fallback_cost, str):
                    try:
                        return float(fallback_cost)
                    except ValueError:
                        verbose_logger.exception(
                            f"litellm.litellm_core_utils.llm_cost_calc.utils.py::_get_cost_per_unit(): Exception occured - {fallback_cost}\nDefaulting to 0.0"
                        )
                break  # Only try the first matching suffix

    return default_value


def calculate_cache_writing_cost(
    cache_creation_tokens: int,
    cache_creation_token_details: Optional[CacheCreationTokenDetails],
    cache_creation_cost_above_1hr: float,
    cache_creation_cost: float,
) -> float:
    """
    Adjust cost of cache creation tokens based on the cache creation token details.
    """
    total_cost: float = 0.0
    if cache_creation_token_details is not None:
        # get the number of 5m and 1h cache creation tokens
        cache_creation_tokens_5m = (
            cache_creation_token_details.ephemeral_5m_input_tokens
        )
        cache_creation_tokens_1h = (
            cache_creation_token_details.ephemeral_1h_input_tokens
        )
        # add the number of 5m and 1h cache creation tokens to the cache creation tokens
        total_cost += (
            cache_creation_tokens_5m * cache_creation_cost
            if cache_creation_tokens_5m is not None
            else 0.0
        )
        total_cost += (
            cache_creation_tokens_1h * cache_creation_cost_above_1hr
            if cache_creation_tokens_1h is not None
            else 0.0
        )
    else:
        total_cost += cache_creation_tokens * cache_creation_cost
    return total_cost


class PromptTokensDetailsResult(TypedDict):
    cache_hit_tokens: int
    cache_creation_tokens: int
    cache_creation_token_details: Optional[CacheCreationTokenDetails]
    text_tokens: int
    audio_tokens: int
    character_count: int
    image_count: int
    video_length_seconds: int


def _parse_prompt_tokens_details(usage: Usage) -> PromptTokensDetailsResult:
    cache_hit_tokens = (
        cast(Optional[int], getattr(usage.prompt_tokens_details, "cached_tokens", 0))
        or 0
    )
    cache_creation_tokens = (
        cast(
            Optional[int],
            getattr(usage.prompt_tokens_details, "cache_creation_tokens", 0),
        )
        or 0
    )
    cache_creation_token_details = (
        cast(
            Optional[CacheCreationTokenDetails],
            getattr(usage.prompt_tokens_details, "cache_creation_token_details", None),
        )
        or None
    )
    text_tokens = (
        cast(Optional[int], getattr(usage.prompt_tokens_details, "text_tokens", None))
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
        cast(Optional[int], getattr(usage.prompt_tokens_details, "image_count", 0)) or 0
    )
    video_length_seconds = (
        cast(
            Optional[int],
            getattr(usage.prompt_tokens_details, "video_length_seconds", 0),
        )
        or 0
    )

    return PromptTokensDetailsResult(
        cache_hit_tokens=cache_hit_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        character_count=character_count,
        image_count=image_count,
        video_length_seconds=video_length_seconds,
    )


class CompletionTokensDetailsResult(TypedDict):
    audio_tokens: int
    text_tokens: int
    reasoning_tokens: int


def _parse_completion_tokens_details(usage: Usage) -> CompletionTokensDetailsResult:
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

    return CompletionTokensDetailsResult(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _calculate_input_cost(
    prompt_tokens_details: PromptTokensDetailsResult,
    model_info: ModelInfo,
    prompt_base_cost: float,
    cache_read_cost: float,
    cache_creation_cost: float,
    cache_creation_cost_above_1hr: float,
) -> float:
    """
    Calculates the input cost for a given model, prompt tokens, and completion tokens.
    """
    prompt_cost = float(prompt_tokens_details["text_tokens"]) * prompt_base_cost

    ### CACHE READ COST - Now uses tiered pricing
    prompt_cost += float(prompt_tokens_details["cache_hit_tokens"]) * cache_read_cost

    ### AUDIO COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_audio_token", prompt_tokens_details["audio_tokens"]
    )

    ### CACHE WRITING COST - Now uses tiered pricing
    prompt_cost += calculate_cache_writing_cost(
        cache_creation_tokens=prompt_tokens_details["cache_creation_tokens"],
        cache_creation_token_details=prompt_tokens_details[
            "cache_creation_token_details"
        ],
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost,
    )

    ### CHARACTER COST

    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_character", prompt_tokens_details["character_count"]
    )

    ### IMAGE COUNT COST
    prompt_cost += calculate_cost_component(
        model_info, "input_cost_per_image", prompt_tokens_details["image_count"]
    )

    ### VIDEO LENGTH COST
    prompt_cost += calculate_cost_component(
        model_info,
        "input_cost_per_video_per_second",
        prompt_tokens_details["video_length_seconds"],
    )

    return prompt_cost


def generic_cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str,
    service_tier: Optional[str] = None,
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
    prompt_tokens_details = PromptTokensDetailsResult(
        cache_hit_tokens=0,
        cache_creation_tokens=0,
        cache_creation_token_details=None,
        text_tokens=usage.prompt_tokens,
        audio_tokens=0,
        character_count=0,
        image_count=0,
        video_length_seconds=0,
    )
    if usage.prompt_tokens_details:
        prompt_tokens_details = _parse_prompt_tokens_details(usage)

    ## EDGE CASE - text tokens not set inside PromptTokensDetails

    if prompt_tokens_details["text_tokens"] == 0:
        text_tokens = (
            usage.prompt_tokens
            - prompt_tokens_details["cache_hit_tokens"]
            - prompt_tokens_details["audio_tokens"]
            - prompt_tokens_details["cache_creation_tokens"]
        )
        prompt_tokens_details["text_tokens"] = text_tokens

    (
        prompt_base_cost,
        completion_base_cost,
        cache_creation_cost,
        cache_creation_cost_above_1hr,
        cache_read_cost,
    ) = _get_token_base_cost(
        model_info=model_info, usage=usage, service_tier=service_tier
    )

    prompt_cost = _calculate_input_cost(
        prompt_tokens_details=prompt_tokens_details,
        model_info=model_info,
        prompt_base_cost=prompt_base_cost,
        cache_read_cost=cache_read_cost,
        cache_creation_cost=cache_creation_cost,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
    )

    ## CALCULATE OUTPUT COST
    text_tokens = 0
    audio_tokens = 0
    reasoning_tokens = 0
    is_text_tokens_total = False
    if usage.completion_tokens_details is not None:
        completion_tokens_details = _parse_completion_tokens_details(usage)
        audio_tokens = completion_tokens_details["audio_tokens"]
        text_tokens = completion_tokens_details["text_tokens"]
        reasoning_tokens = completion_tokens_details["reasoning_tokens"]

    if text_tokens == 0:
        text_tokens = usage.completion_tokens
    if text_tokens == usage.completion_tokens:
        is_text_tokens_total = True
    ## TEXT COST
    completion_cost = float(text_tokens) * completion_base_cost

    _output_cost_per_audio_token = _get_cost_per_unit(
        model_info, "output_cost_per_audio_token", None
    )
    _output_cost_per_reasoning_token = _get_cost_per_unit(
        model_info, "output_cost_per_reasoning_token", None
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


class CostCalculatorUtils:
    @staticmethod
    def _call_type_has_image_response(call_type: str) -> bool:
        """
        Returns True if the call type has an image response

        eg calls that have image response:
        - Image Generation
        - Image Edit
        - Passthrough Image Generation
        """
        if call_type in [
            # image generation
            CallTypes.image_generation.value,
            CallTypes.aimage_generation.value,
            # passthrough image generation
            PassthroughCallTypes.passthrough_image_generation.value,
            # image edit
            CallTypes.image_edit.value,
            CallTypes.aimage_edit.value,
        ]:
            return True
        return False

    @staticmethod
    def route_image_generation_cost_calculator(
        model: str,
        completion_response: ImageResponse,
        custom_llm_provider: Optional[str] = None,
        quality: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        optional_params: Optional[dict] = None,
        call_type: Optional[str] = None,
    ) -> float:
        """
        Route the image generation cost calculator based on the custom_llm_provider
        """
        from litellm.cost_calculator import default_image_cost_calculator
        from litellm.llms.azure_ai.image_generation.cost_calculator import (
            cost_calculator as azure_ai_image_cost_calculator,
        )
        from litellm.llms.bedrock.image.cost_calculator import (
            cost_calculator as bedrock_image_cost_calculator,
        )
        from litellm.llms.gemini.image_generation.cost_calculator import (
            cost_calculator as gemini_image_cost_calculator,
        )
        from litellm.llms.recraft.cost_calculator import (
            cost_calculator as recraft_image_cost_calculator,
        )
        from litellm.llms.vertex_ai.image_generation.cost_calculator import (
            cost_calculator as vertex_ai_image_cost_calculator,
        )

        if size is None:
            size = completion_response.size or "1024-x-1024"
        if quality is None:
            quality = completion_response.quality or "standard"
        if n is None:
            n = len(completion_response.data) if completion_response.data else 0

        if custom_llm_provider == litellm.LlmProviders.VERTEX_AI.value:
            if isinstance(completion_response, ImageResponse):
                return vertex_ai_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                )
        elif custom_llm_provider == litellm.LlmProviders.BEDROCK.value:
            if isinstance(completion_response, ImageResponse):
                return bedrock_image_cost_calculator(
                    model=model,
                    size=size,
                    image_response=completion_response,
                    optional_params=optional_params,
                )
            raise TypeError(
                "completion_response must be of type ImageResponse for bedrock image cost calculation"
            )
        elif custom_llm_provider == litellm.LlmProviders.RECRAFT.value:
            from litellm.llms.recraft.cost_calculator import (
                cost_calculator as recraft_image_cost_calculator,
            )

            return recraft_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.AIML.value:
            from litellm.llms.aiml.image_generation.cost_calculator import (
                cost_calculator as aiml_image_cost_calculator,
            )

            return aiml_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.COMETAPI.value:
            from litellm.llms.cometapi.image_generation.cost_calculator import (
                cost_calculator as cometapi_image_cost_calculator,
            )

            return cometapi_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.GEMINI.value:
            if call_type in (
                CallTypes.image_edit.value,
                CallTypes.aimage_edit.value,
            ):
                from litellm.llms.gemini.image_edit.cost_calculator import (
                    cost_calculator as gemini_image_edit_cost_calculator,
                )

                return gemini_image_edit_cost_calculator(
                    model=model,
                    image_response=completion_response,
                )
            from litellm.llms.gemini.image_generation.cost_calculator import (
                cost_calculator as gemini_image_cost_calculator,
            )

            return gemini_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.AZURE_AI.value:
            return azure_ai_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.FAL_AI.value:
            from litellm.llms.fal_ai.cost_calculator import (
                cost_calculator as fal_ai_image_cost_calculator,
            )

            return fal_ai_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        elif custom_llm_provider == litellm.LlmProviders.RUNWAYML.value:
            from litellm.llms.runwayml.cost_calculator import (
                cost_calculator as runwayml_image_cost_calculator,
            )

            return runwayml_image_cost_calculator(
                model=model,
                image_response=completion_response,
            )
        else:
            return default_image_cost_calculator(
                model=model,
                quality=quality,
                custom_llm_provider=custom_llm_provider,
                n=n,
                size=size,
                optional_params=optional_params,
            )
        return 0.0
