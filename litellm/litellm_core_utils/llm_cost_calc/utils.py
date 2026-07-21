# What is this?
## Helper utilities for cost_per_token()

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, TypedDict, cast

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CallTypes,
    CompletionTokensDetailsWrapper,
    DataResidency,
    ImageResponse,
    ModelInfo,
    PassthroughCallTypes,
    PromptTokensDetailsWrapper,
    ServiceTier,
    Usage,
)
from litellm.utils import get_model_info

# Pre-resolved CallTypes enum values for fast membership checks
_IMAGE_RESPONSE_CALL_TYPES = frozenset(
    {
        CallTypes.image_generation.value,
        CallTypes.aimage_generation.value,
        PassthroughCallTypes.passthrough_image_generation.value,
        CallTypes.image_edit.value,
        CallTypes.aimage_edit.value,
    }
)

# Pre-resolved DataResidency enum values for fast membership checks
_VALID_DATA_RESIDENCIES = frozenset(r.value for r in DataResidency)

# Pre-resolved service-tier cost-key suffixes (e.g. "_priority"). Used per
# request in the cost-calc path, so the f-strings are built once here instead
# of being rebuilt for every model_info key on every call.
_SERVICE_TIER_SUFFIXES: tuple[str, ...] = tuple(f"_{st.value}" for st in ServiceTier)


def _get_token_detail_value(details: object, key: str) -> Optional[int]:
    if isinstance(details, dict):
        value = details.get(key)
    else:
        value = getattr(details, key, None)
    return value if isinstance(value, int) else None


def _get_web_search_requests(server_tool_use: Any) -> Optional[int]:
    """
    Tolerantly read ``web_search_requests`` from a ``server_tool_use`` value
    that may be ``None``, a ``dict``, a ``ServerToolUse`` pydantic instance,
    or any other object supporting attribute access.

    Returns ``None`` when the value cannot be resolved — callers can
    distinguish "absent" from "zero" using ``is None``.

    See https://github.com/BerriAI/litellm/issues/26153 — ``stream_chunk_builder``
    historically left this as a plain ``dict``, which broke direct attribute
    access in cost calculation.
    """
    if server_tool_use is None:
        return None
    if isinstance(server_tool_use, dict):
        return server_tool_use.get("web_search_requests")
    return getattr(server_tool_use, "web_search_requests", None)


def _is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
        return True
    return False


def get_billable_input_tokens(usage: Usage) -> int:
    """
    Returns the number of billable input tokens.
    Subtracts cached tokens from prompt tokens if applicable.
    """
    details = _parse_prompt_tokens_details(usage)
    return usage.prompt_tokens - details["cache_hit_tokens"]


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
    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## CALCULATE INPUT COST
    try:
        if custom_prompt_cost is None:
            assert "input_cost_per_character" in model_info and model_info["input_cost_per_character"] is not None, (
                "model info for model={} does not have 'input_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
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
            assert "output_cost_per_character" in model_info and model_info["output_cost_per_character"] is not None, (
                "model info for model={} does not have 'output_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
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


def _parse_above_token_threshold(key: str) -> float:
    threshold_str = key.split("_above_")[1].split("_tokens")[0]
    return float(threshold_str.replace("k", "")) * (1000 if "k" in threshold_str else 1)


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
    cache_creation_cost_key = _get_service_tier_cost_key("cache_creation_input_token_cost", service_tier)
    cache_read_cost_key = _get_service_tier_cost_key("cache_read_input_token_cost", service_tier)

    prompt_base_cost = cast(float, _get_cost_per_unit(model_info, input_cost_key))
    completion_base_cost = cast(float, _get_cost_per_unit(model_info, output_cost_key))

    # For image generation models that don't have output_cost_per_token,
    # use output_cost_per_image_token as the base cost (all output tokens are image tokens)
    if completion_base_cost == 0.0 or completion_base_cost is None:
        output_image_cost = _get_cost_per_unit(model_info, "output_cost_per_image_token", None)
        if output_image_cost is not None:
            completion_base_cost = cast(float, output_image_cost)
    cache_creation_cost = cast(float, _get_cost_per_unit(model_info, cache_creation_cost_key))
    cache_creation_cost_above_1hr = cast(
        float,
        _get_cost_per_unit(model_info, "cache_creation_input_token_cost_above_1hr"),
    )
    cache_read_cost = cast(float, _get_cost_per_unit(model_info, cache_read_cost_key))

    ## CHECK IF ABOVE THRESHOLD
    # Optimization: collect threshold keys first to avoid sorting all model_info keys.
    # Most models don't have threshold pricing, so we can return early.
    # Exclude service_tier-specific variants (e.g. input_cost_per_token_above_200k_tokens_priority)
    # so that the threshold detection loop only processes standard keys.  The
    # service_tier-specific above-threshold key is resolved later via _get_service_tier_cost_key.
    threshold_keys = [
        k for k in model_info if k.startswith("input_cost_per_token_above_") and not k.endswith(_SERVICE_TIER_SUFFIXES)
    ]
    if not threshold_keys:
        return (
            prompt_base_cost,
            completion_base_cost,
            cache_creation_cost,
            cache_creation_cost_above_1hr,
            cache_read_cost,
        )

    # Only sort the threshold keys (typically 1-2 keys instead of 66+)
    threshold: Optional[float] = None
    for key in sorted(threshold_keys, key=_parse_above_token_threshold, reverse=True):
        value = model_info.get(key)
        if value is not None:
            try:
                # Handle both formats: _above_128k_tokens and _above_128_tokens
                threshold_str = key.split("_above_")[1].split("_tokens")[0]
                threshold = _parse_above_token_threshold(key)
                if usage.prompt_tokens > threshold:
                    # Prefer a service_tier-specific above-threshold key when available,
                    # e.g. input_cost_per_token_priority_above_200k_tokens for Gemini
                    # ON_DEMAND_PRIORITY.  Falls back to the standard key automatically
                    # via _get_cost_per_unit's service_tier fallback logic.
                    tiered_input_key = (
                        _get_service_tier_cost_key(
                            f"input_cost_per_token_above_{threshold_str}_tokens",
                            service_tier,
                        )
                        if service_tier
                        else key
                    )
                    prompt_base_cost = cast(
                        float,
                        _get_cost_per_unit(model_info, tiered_input_key, prompt_base_cost),
                    )
                    tiered_output_key = (
                        _get_service_tier_cost_key(
                            f"output_cost_per_token_above_{threshold_str}_tokens",
                            service_tier,
                        )
                        if service_tier
                        else f"output_cost_per_token_above_{threshold_str}_tokens"
                    )
                    completion_base_cost = cast(
                        float,
                        _get_cost_per_unit(
                            model_info,
                            tiered_output_key,
                            completion_base_cost,
                        ),
                    )

                    # Apply tiered pricing to cache costs
                    cache_creation_tiered_key = (
                        _get_service_tier_cost_key(
                            f"cache_creation_input_token_cost_above_{threshold_str}_tokens",
                            service_tier,
                        )
                        if service_tier
                        else f"cache_creation_input_token_cost_above_{threshold_str}_tokens"
                    )
                    cache_creation_1hr_tiered_key = (
                        _get_service_tier_cost_key(
                            f"cache_creation_input_token_cost_above_1hr_above_{threshold_str}_tokens",
                            service_tier,
                        )
                        if service_tier
                        else f"cache_creation_input_token_cost_above_1hr_above_{threshold_str}_tokens"
                    )
                    cache_read_tiered_key = (
                        _get_service_tier_cost_key(
                            f"cache_read_input_token_cost_above_{threshold_str}_tokens",
                            service_tier,
                        )
                        if service_tier
                        else f"cache_read_input_token_cost_above_{threshold_str}_tokens"
                    )

                    cache_creation_cost = cast(
                        float,
                        _get_cost_per_unit(
                            model_info,
                            cache_creation_tiered_key,
                            cache_creation_cost,
                        ),
                    )

                    cache_creation_cost_above_1hr = cast(
                        float,
                        _get_cost_per_unit(
                            model_info,
                            cache_creation_1hr_tiered_key,
                            cache_creation_cost_above_1hr,
                        ),
                    )

                    cache_read_cost = cast(
                        float,
                        _get_cost_per_unit(model_info, cache_read_tiered_key, cache_read_cost),
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


def calculate_cost_component(model_info: ModelInfo, cost_key: str, usage_value: Optional[float]) -> float:
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
    if cost_per_unit is not None and isinstance(cost_per_unit, float) and usage_value is not None and usage_value > 0:
        return float(usage_value) * cost_per_unit
    return 0.0


def _get_cost_per_unit(model_info: ModelInfo, cost_key: str, default_value: Optional[float] = 0.0) -> Optional[float]:
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
        # Check if any service tier suffix exists in the cost key
        for suffix in _SERVICE_TIER_SUFFIXES:
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
        cache_creation_tokens_5m = cache_creation_token_details.ephemeral_5m_input_tokens
        cache_creation_tokens_1h = cache_creation_token_details.ephemeral_1h_input_tokens
        # add the number of 5m and 1h cache creation tokens to the cache creation tokens
        total_cost += cache_creation_tokens_5m * cache_creation_cost if cache_creation_tokens_5m is not None else 0.0
        total_cost += (
            cache_creation_tokens_1h * cache_creation_cost_above_1hr if cache_creation_tokens_1h is not None else 0.0
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
    image_tokens: int
    video_tokens: int
    character_count: int
    image_count: int
    video_length_seconds: float
    audio_length_seconds: float


def _parse_prompt_tokens_details(usage: Usage) -> PromptTokensDetailsResult:
    cache_hit_tokens = cast(Optional[int], getattr(usage.prompt_tokens_details, "cached_tokens", 0)) or 0
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
    audio_tokens = cast(Optional[int], getattr(usage.prompt_tokens_details, "audio_tokens", 0)) or 0
    image_tokens = cast(Optional[int], getattr(usage.prompt_tokens_details, "image_tokens", 0)) or 0
    video_tokens = _coerce_token_count(getattr(usage.prompt_tokens_details, "video_tokens", 0))
    character_count = (
        cast(
            Optional[int],
            getattr(usage.prompt_tokens_details, "character_count", 0),
        )
        or 0
    )
    image_count = cast(Optional[int], getattr(usage.prompt_tokens_details, "image_count", 0)) or 0
    video_length_seconds = (
        cast(
            Optional[float],
            getattr(usage.prompt_tokens_details, "video_length_seconds", 0),
        )
        or 0.0
    )
    audio_length_seconds = (
        cast(
            Optional[float],
            getattr(usage.prompt_tokens_details, "audio_length_seconds", 0),
        )
        or 0.0
    )

    return PromptTokensDetailsResult(
        cache_hit_tokens=cache_hit_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
        character_count=character_count,
        image_count=image_count,
        video_length_seconds=float(video_length_seconds),
        audio_length_seconds=float(audio_length_seconds),
    )


class CompletionTokensDetailsResult(TypedDict):
    audio_tokens: int
    text_tokens: int
    reasoning_tokens: int
    image_tokens: int
    video_tokens: int


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
    image_tokens = (
        cast(
            Optional[int],
            getattr(usage.completion_tokens_details, "image_tokens", 0),
        )
        or 0
    )
    video_tokens = _coerce_token_count(getattr(usage.completion_tokens_details, "video_tokens", 0))

    return CompletionTokensDetailsResult(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        reasoning_tokens=reasoning_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
    )


def _calculate_input_cost(
    prompt_tokens_details: PromptTokensDetailsResult,
    model_info: ModelInfo,
    prompt_base_cost: float,
    cache_read_cost: float,
    cache_creation_cost: float,
    cache_creation_cost_above_1hr: float,
    service_tier: Optional[str] = None,
) -> float:
    """
    Calculates the input cost for a given model, prompt tokens, and completion tokens.
    """
    prompt_cost = float(prompt_tokens_details["text_tokens"]) * prompt_base_cost

    ### CACHE READ COST - Now uses tiered pricing
    prompt_cost += float(prompt_tokens_details["cache_hit_tokens"]) * cache_read_cost

    ### AUDIO COST
    if prompt_tokens_details["audio_tokens"]:
        audio_cost_key = _get_service_tier_cost_key("input_cost_per_audio_token", service_tier)
        if model_info.get(audio_cost_key) is not None:
            prompt_cost += calculate_cost_component(model_info, audio_cost_key, prompt_tokens_details["audio_tokens"])
        elif model_info.get("input_cost_per_audio_per_second") is None:
            prompt_cost += float(prompt_tokens_details["audio_tokens"]) * prompt_base_cost

    ### IMAGE TOKEN COST
    if prompt_tokens_details["image_tokens"]:
        image_token_cost_key = "input_cost_per_image_token"
        if model_info.get(image_token_cost_key) is not None:
            prompt_cost += calculate_cost_component(
                model_info, image_token_cost_key, prompt_tokens_details["image_tokens"]
            )
        else:
            prompt_cost += float(prompt_tokens_details["image_tokens"]) * prompt_base_cost

    ### VIDEO TOKEN COST
    if prompt_tokens_details["video_tokens"]:
        video_token_cost_key = "input_cost_per_video_token"
        if model_info.get(video_token_cost_key) is not None:
            prompt_cost += calculate_cost_component(
                model_info, video_token_cost_key, prompt_tokens_details["video_tokens"]
            )
        elif model_info.get("input_cost_per_video_per_second") is None:
            prompt_cost += float(prompt_tokens_details["video_tokens"]) * prompt_base_cost

    ### CACHE WRITING COST - Now uses tiered pricing
    if (
        prompt_tokens_details["cache_creation_tokens"]
        or prompt_tokens_details["cache_creation_token_details"] is not None
    ):
        prompt_cost += calculate_cache_writing_cost(
            cache_creation_tokens=prompt_tokens_details["cache_creation_tokens"],
            cache_creation_token_details=prompt_tokens_details["cache_creation_token_details"],
            cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
            cache_creation_cost=cache_creation_cost,
        )

    ### CHARACTER COST
    if prompt_tokens_details["character_count"]:
        prompt_cost += calculate_cost_component(
            model_info,
            "input_cost_per_character",
            prompt_tokens_details["character_count"],
        )

    ### IMAGE COUNT COST
    if prompt_tokens_details["image_count"]:
        prompt_cost += calculate_cost_component(
            model_info, "input_cost_per_image", prompt_tokens_details["image_count"]
        )

    ### VIDEO LENGTH COST
    if prompt_tokens_details["video_length_seconds"]:
        prompt_cost += calculate_cost_component(
            model_info,
            "input_cost_per_video_per_second",
            prompt_tokens_details["video_length_seconds"],
        )

    ### AUDIO LENGTH COST
    if prompt_tokens_details["audio_length_seconds"]:
        prompt_cost += calculate_cost_component(
            model_info,
            "input_cost_per_audio_per_second",
            prompt_tokens_details["audio_length_seconds"],
        )

    return prompt_cost


def _get_regional_uplift_multiplier(model_info: ModelInfo, data_residency: Optional[str]) -> float:
    """
    Resolve the per-model regional-processing uplift multiplier for a given
    data-residency region.

    OpenAI applies a flat percentage uplift (e.g. +10%) on all token costs for
    requests served from a regionalized hostname (eu./us.api.openai.com). The
    multiplier is stored on the model entry as
    ``regional_processing_uplift_multiplier_<region>`` (e.g. 1.10).

    Returns 1.0 (no uplift) when ``data_residency`` is ``None`` or when the
    model has no multiplier configured for the given region.
    """
    if data_residency is None:
        return 1.0
    residency = data_residency.lower()
    if residency not in _VALID_DATA_RESIDENCIES:
        return 1.0
    multiplier = model_info.get(f"regional_processing_uplift_multiplier_{residency}")
    if multiplier is None:
        return 1.0
    try:
        return float(cast(float, multiplier))
    except (TypeError, ValueError):
        verbose_logger.exception(
            "Invalid regional_processing_uplift_multiplier_%s for model; defaulting to 1.0",
            residency,
        )
        return 1.0


def generic_cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str,
    service_tier: Optional[str] = None,
    data_residency: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Handles context caching as well.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - data_residency: optional OpenAI data-residency region (e.g. "eu", "us"),
          used to apply the per-model regional-processing uplift multiplier.

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
        image_tokens=0,
        video_tokens=0,
        character_count=0,
        image_count=0,
        video_length_seconds=0.0,
        audio_length_seconds=0.0,
    )
    if usage.prompt_tokens_details:
        prompt_tokens_details = _parse_prompt_tokens_details(usage)

    ## EDGE CASE - text tokens not set or includes cached tokens (double-counting)
    ## Some providers (like xAI) report text_tokens = prompt_tokens (including cached)
    ## We detect this when: text_tokens + cached_tokens + other > prompt_tokens
    ## Ref: https://github.com/BerriAI/litellm/issues/19680, #14874, #14875

    cache_hit = prompt_tokens_details["cache_hit_tokens"]
    text_tokens = prompt_tokens_details["text_tokens"]
    audio_tokens = prompt_tokens_details["audio_tokens"]
    cache_creation = prompt_tokens_details["cache_creation_tokens"]
    image_tokens = prompt_tokens_details["image_tokens"]
    video_tokens = prompt_tokens_details["video_tokens"]

    # Check for double-counting: sum of details > prompt_tokens means overlap
    total_details = text_tokens + cache_hit + audio_tokens + cache_creation + image_tokens + video_tokens
    has_double_counting = cache_hit > 0 and total_details > usage.prompt_tokens

    if (text_tokens == 0 and prompt_tokens_details["image_count"] == 0) or has_double_counting:
        text_tokens = usage.prompt_tokens - cache_hit - audio_tokens - cache_creation - image_tokens - video_tokens
        # Clamp to zero: inconsistent streaming usage
        if text_tokens < 0:
            text_tokens = 0
        prompt_tokens_details["text_tokens"] = text_tokens

    (
        prompt_base_cost,
        completion_base_cost,
        cache_creation_cost,
        cache_creation_cost_above_1hr,
        cache_read_cost,
    ) = _get_token_base_cost(model_info=model_info, usage=usage, service_tier=service_tier)

    prompt_cost = _calculate_input_cost(
        prompt_tokens_details=prompt_tokens_details,
        model_info=model_info,
        prompt_base_cost=prompt_base_cost,
        cache_read_cost=cache_read_cost,
        cache_creation_cost=cache_creation_cost,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        service_tier=service_tier,
    )

    ## CALCULATE OUTPUT COST
    text_tokens = 0
    audio_tokens = 0
    reasoning_tokens = 0
    image_tokens = 0
    video_tokens = 0
    is_text_tokens_total = False
    if usage.completion_tokens_details is not None:
        completion_tokens_details = _parse_completion_tokens_details(usage)
        audio_tokens = completion_tokens_details["audio_tokens"]
        text_tokens = completion_tokens_details["text_tokens"]
        reasoning_tokens = completion_tokens_details["reasoning_tokens"]
        image_tokens = completion_tokens_details["image_tokens"]
        video_tokens = completion_tokens_details["video_tokens"]

    # Handle text_tokens calculation:
    # 1. If text_tokens is explicitly provided and > 0, use it
    # 2. If there's a breakdown (reasoning/audio/image/video tokens), calculate text_tokens as the remainder
    # 3. If no breakdown at all, assume all completion_tokens are text_tokens
    has_token_breakdown = image_tokens > 0 or audio_tokens > 0 or reasoning_tokens > 0 or video_tokens > 0
    if text_tokens == 0:
        if has_token_breakdown:
            # Calculate text tokens as remainder when we have a breakdown
            # This handles cases like OpenAI's reasoning models where text_tokens isn't provided
            text_tokens = max(
                0,
                usage.completion_tokens - reasoning_tokens - audio_tokens - image_tokens - video_tokens,
            )
        else:
            # No breakdown at all, all tokens are text tokens
            text_tokens = usage.completion_tokens
            is_text_tokens_total = True
    ## TEXT COST
    completion_cost = float(text_tokens) * completion_base_cost

    ## AUDIO COST
    if not is_text_tokens_total and audio_tokens is not None and audio_tokens > 0:
        _output_cost_per_audio_token = _get_cost_per_unit(model_info, "output_cost_per_audio_token", None)
        _output_cost_per_audio_token = (
            _output_cost_per_audio_token if _output_cost_per_audio_token is not None else completion_base_cost
        )
        completion_cost += float(audio_tokens) * _output_cost_per_audio_token

    ## REASONING COST
    if not is_text_tokens_total and reasoning_tokens and reasoning_tokens > 0:
        _output_cost_per_reasoning_token = _get_cost_per_unit(model_info, "output_cost_per_reasoning_token", None)
        _output_cost_per_reasoning_token = (
            _output_cost_per_reasoning_token if _output_cost_per_reasoning_token is not None else completion_base_cost
        )
        completion_cost += float(reasoning_tokens) * _output_cost_per_reasoning_token

    ## IMAGE COST
    if not is_text_tokens_total and image_tokens and image_tokens > 0:
        _output_cost_per_image_token = _get_cost_per_unit(model_info, "output_cost_per_image_token", None)
        _output_cost_per_image_token = (
            _output_cost_per_image_token if _output_cost_per_image_token is not None else completion_base_cost
        )
        completion_cost += float(image_tokens) * _output_cost_per_image_token

    ## VIDEO COST
    if not is_text_tokens_total and video_tokens and video_tokens > 0:
        _output_cost_per_video_token = _get_cost_per_unit(model_info, "output_cost_per_video_token", None)
        _output_cost_per_video_token = (
            _output_cost_per_video_token if _output_cost_per_video_token is not None else completion_base_cost
        )
        completion_cost += float(video_tokens) * _output_cost_per_video_token

    ## REGIONAL DATA-RESIDENCY UPLIFT
    # Applied as a flat multiplier across all token costs for the request
    # when the upstream is a regionalized OpenAI host (eu./us.api.openai.com).
    uplift = _get_regional_uplift_multiplier(model_info, data_residency)
    if uplift != 1.0:
        prompt_cost *= uplift
        completion_cost *= uplift

    return prompt_cost, completion_cost


def _coerce_token_count(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


@dataclass(frozen=True, slots=True)
class TokenTypeCostBreakdown:
    reasoning_cost: float
    cache_read_cost: float
    cache_creation_cost: float


def get_token_type_cost_breakdown(
    model: str,
    custom_llm_provider: Optional[str],
    usage: Usage,
    service_tier: Optional[str] = None,
    data_residency: Optional[str] = None,
) -> TokenTypeCostBreakdown:
    """
    Provider-agnostic cost of reasoning and cache tokens, derived from the usage
    object and model pricing alone.

    This works for every provider, including Perplexity/Cerebras/Dashscope whose
    cost calculators bypass ``generic_cost_per_token``, because cache tokens always
    land on ``prompt_tokens_details`` (via the Usage constructor and provider
    transformations) and reasoning tokens on ``completion_tokens_details``. It reuses
    the same rate-resolution primitives as the total-cost path so the breakdown can
    never drift from the totals. Returns zeros (never raises) when the model or its
    pricing cannot be resolved.
    """
    try:
        model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return TokenTypeCostBreakdown(0.0, 0.0, 0.0)

    (
        _prompt_base_cost,
        completion_base_cost,
        cache_creation_cost_rate,
        cache_creation_cost_above_1hr_rate,
        cache_read_cost_rate,
    ) = _get_token_base_cost(model_info=model_info, usage=usage, service_tier=service_tier)

    reasoning_tokens = (
        _parse_completion_tokens_details(usage)["reasoning_tokens"]
        if usage.completion_tokens_details is not None
        else 0
    )
    if not reasoning_tokens:
        reasoning_tokens = _coerce_token_count(getattr(usage, "reasoning_tokens", 0))

    # Reasoning is billed at the explicit per-reasoning-token rate when the model
    # defines one, otherwise at the standard output-token rate - this mirrors how the
    # total completion cost is computed, so the breakdown can never diverge from it.
    reasoning_rate = _get_cost_per_unit(model_info, "output_cost_per_reasoning_token", None)
    if reasoning_rate is None:
        reasoning_rate = completion_base_cost
    reasoning_cost = float(reasoning_tokens) * reasoning_rate

    cache_read_tokens = 0
    cache_creation_tokens = 0
    cache_creation_token_details: Optional[CacheCreationTokenDetails] = None
    if usage.prompt_tokens_details is not None:
        prompt_tokens_details = _parse_prompt_tokens_details(usage)
        cache_read_tokens = prompt_tokens_details["cache_hit_tokens"]
        cache_creation_tokens = prompt_tokens_details["cache_creation_tokens"]
        cache_creation_token_details = prompt_tokens_details["cache_creation_token_details"]
        # Some OpenAI-compatible providers (e.g. kimi-k2) report cache-write tokens
        # under `cache_write_tokens`; mirror the total-cost normalization path.
        if not cache_creation_tokens:
            cache_creation_tokens = _coerce_token_count(getattr(usage.prompt_tokens_details, "cache_write_tokens", 0))
    # Fall back to the private top-level counters the Usage constructor mirrors cache
    # tokens onto, so providers/callers that bypass prompt_tokens_details are covered.
    if not cache_read_tokens:
        cache_read_tokens = _coerce_token_count(getattr(usage, "_cache_read_input_tokens", 0))
    if not cache_creation_tokens:
        cache_creation_tokens = _coerce_token_count(getattr(usage, "_cache_creation_input_tokens", 0))

    cache_read_cost = float(cache_read_tokens) * cache_read_cost_rate
    cache_creation_cost = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr_rate,
        cache_creation_cost=cache_creation_cost_rate,
    )

    # Apply the same flat regional-processing uplift the totals get, so per-type
    # costs stay reconciled with input_cost/output_cost for regionalized OpenAI hosts.
    uplift = _get_regional_uplift_multiplier(model_info, data_residency)
    if uplift != 1.0:
        reasoning_cost *= uplift
        cache_read_cost *= uplift
        cache_creation_cost *= uplift

    return TokenTypeCostBreakdown(
        reasoning_cost=reasoning_cost,
        cache_read_cost=cache_read_cost,
        cache_creation_cost=cache_creation_cost,
    )


def calculate_image_response_cost_from_usage(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: str,
) -> Optional[float]:
    """
    Calculate image generation cost from usage metadata when available.

    Returns:
        Optional[float]: total cost from token usage, or None when usage metadata
        is missing/incomplete and caller should fall back to flat per-image pricing.
    """
    usage = image_response.usage
    if usage is None:
        return None

    prompt_tokens = usage.input_tokens
    completion_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None

    # ImageResponse may carry a default zeroed usage object even when provider
    # usage metadata is absent. Treat this as missing usage and fall back.
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    input_tokens_details = getattr(usage, "input_tokens_details", None)
    prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
    if input_tokens_details is not None:
        prompt_tokens_details = PromptTokensDetailsWrapper(
            text_tokens=getattr(input_tokens_details, "text_tokens", None),
            image_tokens=getattr(input_tokens_details, "image_tokens", None),
            cached_tokens=0,
        )

    output_tokens_details = getattr(usage, "completion_tokens_details", None)
    if output_tokens_details is None:
        output_tokens_details = getattr(usage, "output_tokens_details", None)

    if output_tokens_details is None:
        completion_tokens_details = CompletionTokensDetailsWrapper(
            text_tokens=0,
            image_tokens=completion_tokens,
            reasoning_tokens=0,
            audio_tokens=0,
        )
    else:
        text_tokens = _get_token_detail_value(output_tokens_details, "text_tokens") or 0
        image_tokens = _get_token_detail_value(output_tokens_details, "image_tokens") or 0
        audio_tokens = _get_token_detail_value(output_tokens_details, "audio_tokens") or 0
        reasoning_tokens = _get_token_detail_value(output_tokens_details, "reasoning_tokens") or 0
        known_output_tokens = text_tokens + image_tokens + audio_tokens + reasoning_tokens
        if completion_tokens > known_output_tokens:
            text_tokens += completion_tokens - known_output_tokens

        completion_tokens_details = CompletionTokensDetailsWrapper(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            reasoning_tokens=reasoning_tokens,
            audio_tokens=audio_tokens,
        )

    normalized_usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=normalized_usage,
        custom_llm_provider=custom_llm_provider,
    )
    return prompt_cost + completion_cost


def calculate_image_response_web_search_cost(
    image_response: ImageResponse,
    custom_llm_provider: str,
    model_info: ModelInfo,
) -> float:
    """
    Cost of Google Search grounding performed during image generation.

    The grounding request count is carried on the image usage object by the
    provider transformers; it is billed with the same per-request accounting
    used for chat completions.
    """
    usage = image_response.usage
    if usage is None:
        return 0.0

    web_search_requests = getattr(usage, "web_search_requests", None)
    if not web_search_requests:
        return 0.0

    from litellm.llms import get_cost_for_web_search_request

    synthetic_usage = Usage(prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=web_search_requests))
    return (
        get_cost_for_web_search_request(
            custom_llm_provider=custom_llm_provider,
            usage=synthetic_usage,
            model_info=model_info,
        )
        or 0.0
    )


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
        return call_type in _IMAGE_RESPONSE_CALL_TYPES

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
        from litellm.llms.bedrock.image_generation.cost_calculator import (
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
            raise TypeError("completion_response must be of type ImageResponse for bedrock image cost calculation")
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
        elif custom_llm_provider == litellm.LlmProviders.OPENAI.value:
            # gpt-image models use token-based pricing.
            model_lower = model.lower()
            if "gpt-image" in model_lower:
                from litellm.llms.openai.image_generation.cost_calculator import (
                    cost_calculator as openai_gpt_image_cost_calculator,
                )

                return openai_gpt_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                    custom_llm_provider=custom_llm_provider,
                )
            # Fall through to default for DALL-E models
            return default_image_cost_calculator(
                model=model,
                quality=quality,
                custom_llm_provider=custom_llm_provider,
                n=n,
                size=size,
                optional_params=optional_params,
            )
        elif custom_llm_provider == litellm.LlmProviders.AZURE.value:
            # gpt-image models use token-based pricing.
            model_lower = model.lower()
            if "gpt-image" in model_lower:
                from litellm.llms.openai.image_generation.cost_calculator import (
                    cost_calculator as openai_gpt_image_cost_calculator,
                )

                return openai_gpt_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                    custom_llm_provider=custom_llm_provider,
                )
            # Fall through to default for DALL-E models
            return default_image_cost_calculator(
                model=model,
                quality=quality,
                custom_llm_provider=custom_llm_provider,
                n=n,
                size=size,
                optional_params=optional_params,
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
