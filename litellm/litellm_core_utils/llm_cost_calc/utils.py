# What is this?
## Helper utilities for cost_per_token()

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, Protocol, SupportsFloat, SupportsIndex, TypedDict, cast

from pydantic import TypeAdapter
from typing_extensions import Buffer, ReadOnly

import litellm
from litellm._internal_context import current_billing_time
from litellm._logging import verbose_logger
from litellm.rust_bridge._native import NativeCallContext, NativePrices, NativeQuote, NativeTokenUsage
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CallTypes,
    CompletionTokensDetailsWrapper,
    CostPerToken,
    CustomPricingLiteLLMParams,
    DataResidency,
    ImageResponse,
    ModelInfo,
    ModelInfoBase,
    PassthroughCallTypes,
    PromptTokensDetailsWrapper,
    ServiceTier,
    Usage,
    text_tokens_without_nested_reasoning,
)
from litellm.utils import get_model_info

# Pre-resolved CallTypes enum values for fast membership checks
_IMAGE_RESPONSE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.image_generation.value,
        CallTypes.aimage_generation.value,
        PassthroughCallTypes.passthrough_image_generation.value,
        CallTypes.image_edit.value,
        CallTypes.aimage_edit.value,
    }
)

_VALID_DATA_RESIDENCIES: Final = frozenset(r.value for r in DataResidency)

_DEPLOYMENT_PRICING_KEYS: Final[frozenset[str]] = frozenset(CustomPricingLiteLLMParams.model_fields)
_IMAGE_REQUEST_PARAMS: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)

_BASE_CACHE_WRITE_RATE_KEY: Final = "cache_creation_input_token_cost"

_CACHE_WRITE_RATE_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "5m": _BASE_CACHE_WRITE_RATE_KEY,
        "30m": _BASE_CACHE_WRITE_RATE_KEY,
        "1hr": "cache_creation_input_token_cost_above_1hr",
        "12hr": "cache_creation_input_token_cost_above_12hr",
    }
)

_TTL_PRICED_CACHE_WRITE_RATE_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {ttl: key for ttl, key in _CACHE_WRITE_RATE_KEYS.items() if key != _BASE_CACHE_WRITE_RATE_KEY}
)


_IMAGE_TOKEN_RATE_KEYS: Final[tuple[str, ...]] = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_image_token",
    "output_cost_per_image_token",
)


def _deployment_prices_images_without_an_output_price(pricing: ModelInfo | None) -> bool:
    return pricing is not None and (
        "input_cost_per_pixel" in pricing
        or ("input_cost_per_image" in pricing and "output_cost_per_image" not in pricing)
    )


# Pre-resolved service-tier cost-key suffixes (e.g. "_priority"). Used per
# request in the cost-calc path, so the f-strings are built once here instead
# of being rebuilt for every model_info key on every call. Longest-first so a
# substring match resolves "_ultrafast" before "_fast".
_SERVICE_TIER_SUFFIXES: Final[tuple[str, ...]] = tuple(
    sorted((f"_{st.value}" for st in ServiceTier), key=len, reverse=True)
)

_SERVICE_TIER_TO_COST_KEY_SUFFIX: Final[Mapping[str, str]] = MappingProxyType(
    {
        ServiceTier.FLEX.value: ServiceTier.FLEX.value,
        ServiceTier.BALANCED.value: ServiceTier.BALANCED.value,
        ServiceTier.PRIORITY.value: ServiceTier.PRIORITY.value,
        ServiceTier.FAST.value: ServiceTier.PRIORITY.value,
        ServiceTier.ULTRAFAST.value: ServiceTier.ULTRAFAST.value,
    }
)
SERVICE_TIER_COST_KEY_SUFFIXES: Final[tuple[str, ...]] = tuple(
    sorted(frozenset(f"_{suffix}" for suffix in _SERVICE_TIER_TO_COST_KEY_SUFFIX.values()))
)

_INCLUSIVE_THRESHOLD_PROVIDERS: Final = frozenset({"xai"})
_BATCH_KEY_SUFFIX: Final = "_batches"
_BATCH_RATE_PREFIXES: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_read_input_image_token_cost",
    "cache_creation_input_token_cost",
)
_BATCH_TIER_PREFIXES: Final = (*_BATCH_RATE_PREFIXES, *_TTL_PRICED_CACHE_WRITE_RATE_KEYS.values())
_BATCH_TIER_KEY: Final = re.compile(rf"^({'|'.join(_BATCH_TIER_PREFIXES)})_above_(\d+k?)_tokens{_BATCH_KEY_SUFFIX}$")
_NON_STANDARD_THRESHOLD_SUFFIXES: Final = (*_SERVICE_TIER_SUFFIXES, _BATCH_KEY_SUFFIX)


def _get_token_detail_value(details: object, key: str) -> int | None:
    if isinstance(details, dict):
        value = details.get(key)
    else:
        value = getattr(details, key, None)
    return value if isinstance(value, int) else None


_IMAGE_SIZE_PATTERN: Final = re.compile(r"\d+(?:x|-x-|\*)\d+")


def _requested_image_param(optional_params: Mapping[str, object] | None, key: str) -> str | None:
    value: Final = None if optional_params is None else optional_params.get(key)
    return value if isinstance(value, str) else None


def _requested_image_size(optional_params: Mapping[str, object] | None) -> str | None:
    value: Final = _requested_image_param(optional_params, "size")
    return value.replace("*", "x") if value is not None and _IMAGE_SIZE_PATTERN.fullmatch(value) else None


def get_web_search_requests(server_tool_use: object) -> int | None:
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


def get_web_search_requests_from_usage(usage: Usage) -> int | None:
    """Read ``web_search_requests`` from a ``Usage``'s ``server_tool_use``.

    ``Usage`` deletes unset optional fields from ``__dict__`` (see
    ``SafeAttributeModel``), so direct attribute access can raise
    ``AttributeError``; ``getattr`` with a default is required here.
    """
    return get_web_search_requests(getattr(usage, "server_tool_use", None))


def is_above_128k(tokens: float) -> bool:
    if tokens > 128000:
        return True
    return False


_is_above_128k = is_above_128k


def get_billable_input_tokens(usage: Usage) -> int:
    """
    Returns the number of billable input tokens.
    Subtracts cached tokens from prompt tokens if applicable.
    """
    details: Final = parse_prompt_tokens_details(usage)
    return usage.prompt_tokens - details["cache_hit_tokens"]


def select_cost_metric_for_model(
    model_info: ModelInfoBase,
) -> Literal["cost_per_character", "cost_per_token"]:
    """
    Select 'cost_per_character' if model_info has 'input_cost_per_character'
    Select 'cost_per_token' if model_info has 'input_cost_per_token'
    """
    if model_info.get("input_cost_per_character") is not None:
        return "cost_per_character"
    elif model_info.get("input_cost_per_token") is not None:
        return "cost_per_token"
    else:
        raise ValueError(
            f"Model {model_info['key']} does not have 'input_cost_per_character' or 'input_cost_per_token'"
        )


def generic_cost_per_character(
    model: str,
    custom_llm_provider: str,
    prompt_characters: float,
    completion_characters: float,
    custom_prompt_cost: float | None,
    custom_completion_cost: float | None,
) -> tuple[float | None, float | None]:
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
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## CALCULATE INPUT COST
    try:
        if custom_prompt_cost is None:
            assert "input_cost_per_character" in model_info and model_info["input_cost_per_character"] is not None, (
                f"model info for model={model} does not have 'input_cost_per_character'-pricing\nmodel_info={model_info}"
            )
            custom_prompt_cost = model_info["input_cost_per_character"]

        prompt_cost = prompt_characters * custom_prompt_cost
    except Exception as e:
        verbose_logger.exception(
            "litellm.litellm_core_utils.llm_cost_calc.utils.py::cost_per_character(): Exception occured - %s\nDefaulting to None",
            e,
        )

        prompt_cost = None

    ## CALCULATE OUTPUT COST
    try:
        if custom_completion_cost is None:
            assert "output_cost_per_character" in model_info and model_info["output_cost_per_character"] is not None, (
                f"model info for model={model} does not have 'output_cost_per_character'-pricing\nmodel_info={model_info}"
            )
            custom_completion_cost = model_info["output_cost_per_character"]
        completion_cost = completion_characters * custom_completion_cost
    except Exception as e:
        verbose_logger.exception(
            "litellm.litellm_core_utils.llm_cost_calc.utils.py::cost_per_character(): Exception occured - %s\nDefaulting to None",
            e,
        )

        completion_cost = None

    return prompt_cost, completion_cost


_generic_cost_per_character = generic_cost_per_character


def get_service_tier_cost_key(base_key: str, service_tier: str | None) -> str:
    """
    Get the appropriate cost key based on service tier.

    Args:
        base_key: The base cost key (e.g., "input_cost_per_token")
        service_tier: The service tier ("flex", "balanced", "priority", "fast", "ultrafast", or None for standard)

    Returns:
        str: The cost key to use (e.g., "input_cost_per_token_flex" or "input_cost_per_token")
    """
    if service_tier is None:
        return base_key

    suffix: Final = _SERVICE_TIER_TO_COST_KEY_SUFFIX.get(service_tier.lower())
    if suffix is None:
        return base_key

    return f"{base_key}_{suffix}"


_get_service_tier_cost_key = get_service_tier_cost_key


def _parse_token_threshold(threshold: str) -> float:
    return float(threshold.replace("k", "")) * (1000 if "k" in threshold else 1)


def _parse_above_token_threshold(key: str) -> float:
    return _parse_token_threshold(key.rsplit("_above_", 1)[1].split("_tokens")[0])


def _prompt_exceeds_threshold(prompt_tokens: int, threshold: float, inclusive: bool) -> bool:
    return prompt_tokens > threshold or (inclusive and prompt_tokens == threshold)


@dataclass(frozen=True, slots=True)
class BatchCostRates:
    input: float | None
    output: float | None
    cache_read: float | None
    cache_creation: float | None
    cache_creation_rates_by_ttl: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    cache_read_image: float | None = None


def _batch_rate(model_info: ModelInfo, key: str) -> float | None:
    value: Final = model_info.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _batch_tier_rate(model_info: ModelInfo, tier_key: str, flat_key: str) -> float | None:
    tier_rate: Final = _batch_rate(model_info, tier_key)
    return _batch_rate(model_info, flat_key) if tier_rate is None else tier_rate


def _batch_tier_thresholds(model_info: ModelInfo, prefix: str) -> frozenset[str]:
    return frozenset(
        tier.group(2)
        for key, value in model_info.items()
        if value is not None and (tier := _BATCH_TIER_KEY.match(key)) is not None and tier.group(1) == prefix
    )


def _crossed_batch_tier(model_info: ModelInfo, prefix: str, usage: Usage, inclusive: bool) -> str | None:
    return next(
        (
            threshold
            for threshold in sorted(
                _batch_tier_thresholds(model_info, prefix), key=_parse_token_threshold, reverse=True
            )
            if _prompt_exceeds_threshold(usage.prompt_tokens, _parse_token_threshold(threshold), inclusive)
        ),
        None,
    )


def _batch_rate_for_prefix(model_info: ModelInfo, prefix: str, usage: Usage, inclusive: bool) -> float | None:
    flat_key: Final = f"{prefix}{_BATCH_KEY_SUFFIX}"
    threshold: Final = _crossed_batch_tier(model_info, prefix, usage, inclusive)
    if threshold is None:
        return _batch_rate(model_info, flat_key)
    return _batch_tier_rate(model_info, f"{prefix}_above_{threshold}_tokens{_BATCH_KEY_SUFFIX}", flat_key)


def get_batch_cost_rates(model_info: ModelInfo, usage: Usage, custom_llm_provider: str | None) -> BatchCostRates:
    inclusive: Final = custom_llm_provider in _INCLUSIVE_THRESHOLD_PROVIDERS
    return BatchCostRates(
        input=_batch_rate_for_prefix(model_info, "input_cost_per_token", usage, inclusive),
        output=_batch_rate_for_prefix(model_info, "output_cost_per_token", usage, inclusive),
        cache_read=_batch_rate_for_prefix(model_info, "cache_read_input_token_cost", usage, inclusive),
        cache_creation=_batch_rate_for_prefix(model_info, "cache_creation_input_token_cost", usage, inclusive),
        cache_creation_rates_by_ttl=MappingProxyType(
            {
                ttl: rate
                for ttl, key in _TTL_PRICED_CACHE_WRITE_RATE_KEYS.items()
                if (rate := _batch_rate_for_prefix(model_info, key, usage, inclusive)) is not None
            }
        ),
        cache_read_image=_batch_rate_for_prefix(model_info, "cache_read_input_image_token_cost", usage, inclusive),
    )


def get_cost_per_unit(model_info: ModelInfo, cost_key: str, default_value: float | None = 0.0) -> float | None:
    # Sometimes the cost per unit is a string (e.g.: If a value like "3e-7" was read from the config.yaml)
    cost_per_unit: Final = model_info.get(cost_key)
    if isinstance(cost_per_unit, float):
        return cost_per_unit
    if isinstance(cost_per_unit, int):
        return float(cost_per_unit)
    if isinstance(cost_per_unit, str):
        try:
            return float(cost_per_unit)
        except ValueError:
            verbose_logger.exception(
                "litellm.litellm_core_utils.llm_cost_calc.utils.py::calculate_cost_per_component(): Exception occured - %s\nDefaulting to 0.0",
                cost_per_unit,
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
                            "litellm.litellm_core_utils.llm_cost_calc.utils.py::get_cost_per_unit(): Exception occured - %s\nDefaulting to 0.0",
                            fallback_cost,
                        )
                break  # Only try the first matching suffix

    return default_value


_get_cost_per_unit = get_cost_per_unit


def deployment_pricing(model_info: ModelInfo | None) -> ModelInfo | None:
    """The prices a deployment sets itself, as floats; None when it sets none that parse."""
    if model_info is None:
        return None
    priced_keys: Final = tuple(key for key in _DEPLOYMENT_PRICING_KEYS if model_info.get(key) is not None)
    pricing: Final = MappingProxyType(
        {
            key: price
            for key in priced_keys
            if (price := get_cost_per_unit(model_info, key, default_value=None)) is not None
        }
    )
    if not pricing:
        return None
    return cast(ModelInfo, pricing)  # cast-ok: a read-only subset of ModelInfo pricing keys, values validated above


def prices_tokens(model_info: ModelInfo) -> bool:
    """Whether the price table carries any token rate, so a token-priced calculator can bill from usage."""
    return any(model_info.get(key) is not None for key in _IMAGE_TOKEN_RATE_KEYS)


def flat_image_cost(model_info: ModelInfo | None, image_response: ImageResponse) -> float:
    """The per-image price times the images returned; 0.0 when the table sets no per-image price."""
    if model_info is None:
        return 0.0
    output_cost_per_image: Final = get_cost_per_unit(model_info, "output_cost_per_image", default_value=None) or 0.0
    num_images: Final = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images


def resolve_image_model_info(model: str, custom_llm_provider: str, model_info: ModelInfo | None) -> ModelInfo:
    """The price table an image cost calculator consults for ``model``.

    ``shared_backend_model_info`` keeps deployment prices off the shared ``{provider}/{model}`` key, so
    a name lookup alone reads the public rate, and a model only the deployment prices has no entry at all.
    """
    if model_info is None:
        return get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    try:
        shared_model_info: Final = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return model_info
    resolved: Final[ModelInfo] = {**shared_model_info, **model_info}
    return resolved


def calculate_cache_writing_cost(
    cache_creation_tokens: int,
    cache_creation_token_details: CacheCreationTokenDetails | None,
    cache_creation_cost: float,
    cache_creation_cost_by_ttl: Mapping[str, float],
) -> float:
    """
    Bill cache writes per lifetime: Anthropic reports 5m and 1h counts, OpenAI reports 30m and 12h counts.
    Writes no lifetime count covers, such as a lifetime a provider adds later, bill at the base write rate.
    """
    if cache_creation_token_details is None:
        return cache_creation_tokens * cache_creation_cost
    tokens_by_ttl: Final[Mapping[str, int | None]] = MappingProxyType(
        {
            "5m": cache_creation_token_details.ephemeral_5m_input_tokens,
            "30m": cache_creation_token_details.ephemeral_30m_input_tokens,
            "1hr": cache_creation_token_details.ephemeral_1h_input_tokens,
            "12hr": cache_creation_token_details.ephemeral_12h_input_tokens,
        }
    )
    unattributed_tokens: Final = max(cache_creation_tokens - sum(t or 0 for t in tokens_by_ttl.values()), 0)
    return unattributed_tokens * cache_creation_cost + sum(
        (tokens or 0) * cache_creation_cost_by_ttl.get(ttl, cache_creation_cost)
        for ttl, tokens in tokens_by_ttl.items()
    )


class PromptTokensDetailsResult(TypedDict):
    cache_hit_tokens: int
    cache_hit_audio_tokens: ReadOnly[int]

    cached_text_tokens: ReadOnly[int]
    cached_audio_tokens: ReadOnly[int]
    cached_image_tokens: ReadOnly[int]
    has_cached_tokens_details: ReadOnly[bool]
    cache_creation_tokens: int
    cache_creation_token_details: CacheCreationTokenDetails | None
    text_tokens: int
    audio_tokens: int
    image_tokens: int
    video_tokens: int
    character_count: int
    image_count: int
    video_length_seconds: float
    audio_length_seconds: float
    query_count: int


def parse_prompt_tokens_details(usage: Usage) -> PromptTokensDetailsResult:
    cache_hit_tokens: Final = cast(int | None, getattr(usage.prompt_tokens_details, "cached_tokens", 0)) or 0
    cache_creation_tokens: Final = (
        cast(
            int | None,
            getattr(usage.prompt_tokens_details, "cache_write_tokens", 0)
            or getattr(usage.prompt_tokens_details, "cache_creation_tokens", 0),
        )
        or 0
    )
    cache_creation_token_details: Final = (
        cast(
            CacheCreationTokenDetails | None,
            getattr(usage.prompt_tokens_details, "cache_creation_token_details", None),
        )
        or None
    )
    cached_tokens_details: Final = getattr(usage.prompt_tokens_details, "cached_tokens_details", None)
    cached_audio_tokens: Final = min(
        _get_token_detail_value(cached_tokens_details, "audio_tokens") or 0, cache_hit_tokens
    )
    cached_text_tokens: Final = min(
        _get_token_detail_value(cached_tokens_details, "text_tokens") or 0,
        cache_hit_tokens - cached_audio_tokens,
    )
    cached_image_tokens: Final = min(
        _get_token_detail_value(cached_tokens_details, "image_tokens") or 0,
        cache_hit_tokens - cached_audio_tokens - cached_text_tokens,
    )
    text_tokens: Final = max(
        (
            cast(int | None, getattr(usage.prompt_tokens_details, "text_tokens", None))
            or 0  # default to prompt tokens, if this field is not set
        )
        - cached_text_tokens,
        0,
    )
    audio_tokens: Final = max(
        (cast(int | None, getattr(usage.prompt_tokens_details, "audio_tokens", 0)) or 0) - cached_audio_tokens,
        0,
    )
    image_tokens: Final = max(
        (cast(int | None, getattr(usage.prompt_tokens_details, "image_tokens", 0)) or 0) - cached_image_tokens,
        0,
    )
    video_tokens: Final = _coerce_token_count(getattr(usage.prompt_tokens_details, "video_tokens", 0))
    character_count: Final = (
        cast(
            int | None,
            getattr(usage.prompt_tokens_details, "character_count", 0),
        )
        or 0
    )
    image_count: Final = cast(int | None, getattr(usage.prompt_tokens_details, "image_count", 0)) or 0
    video_length_seconds: Final = (
        cast(
            float | None,
            getattr(usage.prompt_tokens_details, "video_length_seconds", 0),
        )
        or 0.0
    )
    audio_length_seconds: Final = (
        cast(
            float | None,
            getattr(usage.prompt_tokens_details, "audio_length_seconds", 0),
        )
        or 0.0
    )
    query_count: Final = _coerce_token_count(getattr(usage.prompt_tokens_details, "query_count", 0))

    return PromptTokensDetailsResult(
        cache_hit_tokens=cache_hit_tokens,
        cache_hit_audio_tokens=cached_audio_tokens,
        cached_text_tokens=cached_text_tokens,
        cached_audio_tokens=cached_audio_tokens,
        cached_image_tokens=cached_image_tokens,
        has_cached_tokens_details=cached_tokens_details is not None,
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
        query_count=query_count,
    )


class CompletionTokensDetailsResult(TypedDict):
    audio_tokens: int
    text_tokens: int
    reasoning_tokens: int
    image_tokens: int
    video_tokens: int


def parse_completion_tokens_details(usage: Usage) -> CompletionTokensDetailsResult:
    audio_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "audio_tokens", 0),
        )
        or 0
    )
    reported_text_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "text_tokens", None),
        )
        or 0  # default to completion tokens, if this field is not set
    )
    reasoning_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0),
        )
        or 0
    )
    image_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "image_tokens", 0),
        )
        or 0
    )
    video_tokens: Final = _coerce_token_count(getattr(usage.completion_tokens_details, "video_tokens", 0))
    text_tokens: Final = text_tokens_without_nested_reasoning(
        completion_tokens=usage.completion_tokens,
        text_tokens=reported_text_tokens,
        reasoning_tokens=reasoning_tokens,
        other_modality_tokens=audio_tokens + image_tokens + video_tokens,
    )

    return CompletionTokensDetailsResult(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        reasoning_tokens=reasoning_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
    )


def get_modality_token_cost_key(
    model_info: ModelInfo,
    cost_key: str,
    prompt_tokens: int,
    *,
    threshold_is_inclusive: bool = False,
) -> str:
    suffix: Final = next((suffix for suffix in _NON_STANDARD_THRESHOLD_SUFFIXES if cost_key.endswith(suffix)), "")
    base_key: Final = cost_key.removesuffix(suffix)
    threshold_key: Final = max(
        (
            key
            for key in model_info
            if key.startswith(f"{base_key}_above_")
            and key.endswith(("_tokens", f"_tokens{suffix}"))
            and model_info.get(key) is not None
            and _prompt_exceeds_threshold(prompt_tokens, _parse_above_token_threshold(key), threshold_is_inclusive)
        ),
        key=_parse_above_token_threshold,
        default=None,
    )
    if threshold_key is None:
        return cost_key
    return f"{threshold_key.removesuffix(suffix)}{suffix}"


def get_regional_uplift_multiplier(model_info: ModelInfo, data_residency: str | None) -> float:
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
    residency: Final = data_residency.lower()
    if residency not in _VALID_DATA_RESIDENCIES:
        return 1.0
    multiplier: Final[object] = model_info.get(f"regional_processing_uplift_multiplier_{residency}")
    if multiplier is None:
        return 1.0
    if not isinstance(multiplier, (str, Buffer, SupportsFloat, SupportsIndex)):
        verbose_logger.error(
            "Invalid regional_processing_uplift_multiplier_%s for model (got %s); defaulting to 1.0",
            residency,
            type(multiplier).__name__,
        )
        return 1.0
    try:
        return float(multiplier)
    except (TypeError, ValueError):
        verbose_logger.exception(
            "Invalid regional_processing_uplift_multiplier_%s for model; defaulting to 1.0",
            residency,
        )
        return 1.0


_get_regional_uplift_multiplier = get_regional_uplift_multiplier


def get_fedramp_uplift_multiplier(model_info: ModelInfo, data_residency: str | None) -> float:
    """
    Resolve the per-model uplift multiplier for the OpenAI FedRAMP endpoint.

    OpenAI applies a flat percentage uplift (e.g. +10%) on all token costs for
    requests served from gov.api.openai.com. The multiplier is stored on the
    model entry as ``fedramp_uplift_multiplier`` (e.g. 1.10).

    Returns 1.0 (no uplift) unless ``data_residency`` is ``"fedramp"`` and the
    model has a multiplier configured.
    """
    if data_residency is None or data_residency.lower() != DataResidency.FEDRAMP.value:
        return 1.0
    multiplier: Final = model_info.get("fedramp_uplift_multiplier")
    if multiplier is None:
        return 1.0
    try:
        return float(multiplier)
    except (TypeError, ValueError):
        verbose_logger.exception("Invalid fedramp_uplift_multiplier for model; defaulting to 1.0")
        return 1.0


def get_vertex_regional_endpoint_uplift(model_info: Mapping[str, object], vertex_location: str | None) -> float:
    """
    Resolve the per-model uplift multiplier for Vertex AI non-global (regional and
    multi-region) endpoints.

    Google prices every non-global endpoint at a flat premium over the global
    endpoint (e.g. 1.10 = +10%) on all token types for the models that carry
    regional pricing. The multiplier is stored on the model entry as
    ``regional_endpoint_uplift_multiplier``.

    Returns 1.0 (no uplift) when ``vertex_location`` is ``None`` or ``"global"``,
    or when the model has no multiplier configured.
    """
    if vertex_location is None or vertex_location.lower() == "global":
        return 1.0
    multiplier: Final = model_info.get("regional_endpoint_uplift_multiplier")
    if multiplier is None:
        return 1.0
    try:
        return float(cast(float, multiplier))  # cast-ok: pricing multiplier is external model data
    except (TypeError, ValueError):
        verbose_logger.exception(
            "Invalid regional_endpoint_uplift_multiplier for model; defaulting to 1.0",
        )
        return 1.0


def get_provider_specific_geo_multiplier(model_info: ModelInfo, usage: Usage) -> float:
    """
    Resolve the provider-specific regional pricing multiplier for the geo the
    request was served from (``usage.inference_geo``), e.g. Anthropic's ``us: 1.1``
    stored under ``provider_specific_entry``. The regional surcharge applies to
    every token type, so per-type cost breakdowns must scale by it too.

    Returns 1.0 when the request was served globally or the model carries no
    multiplier for the geo.
    """
    inference_geo: Final = getattr(usage, "inference_geo", None)
    if not isinstance(inference_geo, str) or inference_geo.lower() in ("global", "not_available"):
        return 1.0
    provider_specific_entry: Final[dict[str, float]] = model_info.get("provider_specific_entry") or {}
    return float(provider_specific_entry.get(inference_geo.lower(), 1.0))


def get_provider_specific_speed_multiplier(model_info: ModelInfo, usage: Usage) -> float:
    provider_specific_entry: Final = model_info.get("provider_specific_entry")
    if getattr(usage, "speed", None) != "fast" or not provider_specific_entry:
        return 1.0
    return float(provider_specific_entry.get("fast", 1.0))


def _billed_prompt_tokens(usage: Usage) -> PromptTokensDetailsResult:
    """Prompt-side counts with the overlap between cached and per-modality tokens removed.

    Some providers (like xAI) report text_tokens = prompt_tokens, cached tokens included. That shows
    as text + cached + other modalities adding up to more than prompt_tokens. See issues #19680,
    #14874 and #14875. Cached and per-modality counts are both subsets of prompt_tokens and may
    overlap, so a modality only bills what the cache did not already cover.
    """
    if not usage.prompt_tokens_details:
        return PromptTokensDetailsResult(
            cache_hit_tokens=0,
            cache_hit_audio_tokens=0,
            cached_text_tokens=0,
            cached_audio_tokens=0,
            cached_image_tokens=0,
            has_cached_tokens_details=False,
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
            query_count=0,
        )
    details: Final = parse_prompt_tokens_details(usage)
    cache_hit: Final = details["cache_hit_tokens"]
    cache_creation: Final = details["cache_creation_tokens"]
    text_tokens: Final = details["text_tokens"]
    audio_tokens: Final = details["audio_tokens"]
    image_tokens: Final = details["image_tokens"]
    video_tokens: Final = details["video_tokens"]
    total_details: Final = text_tokens + cache_hit + audio_tokens + cache_creation + image_tokens + video_tokens
    if (cache_hit > 0 or cache_creation > 0) and total_details > usage.prompt_tokens:
        uncached_budget: Final = max(usage.prompt_tokens - cache_hit - cache_creation, 0)
        billable_audio: Final = min(audio_tokens, uncached_budget)
        billable_image: Final = min(image_tokens, uncached_budget - billable_audio)
        billable_video: Final = min(video_tokens, uncached_budget - billable_audio - billable_image)
        return {
            **details,
            "audio_tokens": billable_audio,
            "image_tokens": billable_image,
            "video_tokens": billable_video,
            "text_tokens": uncached_budget - billable_audio - billable_image - billable_video,
        }
    if text_tokens == 0 and details["image_count"] == 0:
        # inconsistent streaming usage can report more parts than prompt tokens, so clamp at zero
        remainder: Final = usage.prompt_tokens - cache_hit - audio_tokens - cache_creation - image_tokens - video_tokens
        return {**details, "text_tokens": max(remainder, 0)}
    return details


def _billed_completion_tokens(usage: Usage) -> CompletionTokensDetailsResult:
    """Completion-side counts. Text tokens the provider did not report are whatever the reasoning,
    audio, image and video tokens leave, or every completion token when there is no breakdown."""
    if usage.completion_tokens_details is None:
        return CompletionTokensDetailsResult(
            audio_tokens=0,
            text_tokens=usage.completion_tokens,
            reasoning_tokens=0,
            image_tokens=0,
            video_tokens=0,
        )
    details: Final = parse_completion_tokens_details(usage)
    other_tokens: Final = (
        details["reasoning_tokens"] + details["audio_tokens"] + details["image_tokens"] + details["video_tokens"]
    )
    if details["text_tokens"] != 0:
        return details
    if other_tokens > 0:
        return {**details, "text_tokens": max(0, usage.completion_tokens - other_tokens)}
    return {**details, "text_tokens": usage.completion_tokens}


def _count(value: float | None) -> int:
    return max(int(value or 0), 0)


def _reads_explicit_cache(usage: Usage) -> bool:
    details: Final = usage.prompt_tokens_details
    if details is None or getattr(details, "cache_type", None) != "ephemeral":
        return False
    return parse_prompt_tokens_details(usage)["cache_hit_tokens"] > 0


def billed_cache_read_rate(quote: NativeQuote, usage: Usage) -> float:
    """The rate ``usage``'s cached tokens bill at: the explicit cache-read rate when they are reads of a
    cache the caller created, the implicit one otherwise."""
    return quote.explicit_cache_read if _reads_explicit_cache(usage) else quote.cache_read


def billed_quantities(usage: Usage) -> NativeTokenUsage:
    """What ``usage`` bills on each meter, the counts the provider reported once their overlap is removed."""
    prompt: Final = _billed_prompt_tokens(usage)
    completion: Final = _billed_completion_tokens(usage)
    classified_cached_tokens: Final = (
        prompt["cached_text_tokens"] + prompt["cached_audio_tokens"] + prompt["cached_image_tokens"]
    )
    write_details: Final = prompt["cache_creation_token_details"]
    one_hour_writes: Final = 0 if write_details is None else _count(write_details.ephemeral_1h_input_tokens)
    twelve_hour_writes: Final = 0 if write_details is None else _count(write_details.ephemeral_12h_input_tokens)
    short_lived_writes: Final = (
        0
        if write_details is None
        else _count(write_details.ephemeral_5m_input_tokens) + _count(write_details.ephemeral_30m_input_tokens)
    )
    # writes no lifetime count covers bill at the base write rate, as the short-lived ones do
    unattributed_writes: Final = max(
        prompt["cache_creation_tokens"] - short_lived_writes - one_hour_writes - twelve_hour_writes, 0
    )
    return NativeTokenUsage(
        prompt_tokens=_count(usage.prompt_tokens),
        input_text_tokens=_count(prompt["text_tokens"]),
        input_audio_tokens=_count(prompt["audio_tokens"]),
        input_image_tokens=_count(prompt["image_tokens"]),
        input_video_tokens=_count(prompt["video_tokens"]),
        cached_text_tokens=_count(
            prompt["cached_text_tokens"] + max(prompt["cache_hit_tokens"] - classified_cached_tokens, 0)
        ),
        cached_audio_tokens=_count(prompt["cached_audio_tokens"]),
        cached_image_tokens=_count(prompt["cached_image_tokens"]),
        explicit_cache_read=_reads_explicit_cache(usage),
        cache_write_tokens=_count(unattributed_writes + short_lived_writes),
        cache_write_1h_tokens=one_hour_writes,
        cache_write_12h_tokens=twelve_hour_writes,
        input_characters=_count(prompt["character_count"]),
        input_images=_count(prompt["image_count"]),
        input_audio_seconds=max(float(prompt["audio_length_seconds"]), 0.0),
        input_video_seconds=max(float(prompt["video_length_seconds"]), 0.0),
        queries=_count(prompt["query_count"]),
        output_text_tokens=_count(completion["text_tokens"]),
        output_audio_tokens=_count(completion["audio_tokens"]),
        output_image_tokens=_count(completion["image_tokens"]),
        output_video_tokens=_count(completion["video_tokens"]),
        reasoning_tokens=_count(completion["reasoning_tokens"]),
    )


def call_context(
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    current_time: datetime | None,
) -> NativeCallContext:
    """The call facts that pick a price. A missing ``current_time`` is the pinned billing moment, and a
    naive one is read as UTC."""
    reference: Final = current_time if current_time is not None else current_billing_time()
    return NativeCallContext(
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        at=reference.astimezone(timezone.utc)
        if reference.tzinfo is not None
        else reference.replace(tzinfo=timezone.utc),
    )


def generic_cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str,
    service_tier: str | None = None,
    data_residency: str | None = None,
    model_info: ModelInfo | None = None,
    vertex_location: str | None = None,
    current_time: datetime | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Handles context caching as well.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - data_residency: optional OpenAI data-residency region (e.g. "eu", "us") or
          "fedramp", used to apply the per-model regional-processing or FedRAMP uplift multiplier.
        - model_info: the deployment's effective rates, for a caller that already resolved them.
          A name cannot express a per-deployment override: those are registered under the
          deployment id and kept off the shared model-name key
        - vertex_location: optional Vertex AI location the request was served from
          (e.g. "us-east5", "global"), used to apply the per-model
          regional-endpoint uplift multiplier when non-global.
        - current_time: the moment the request is billed at, for off_peak_pricing; defaults to now, UTC

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    row: Final = (
        get_model_info(model=model, custom_llm_provider=custom_llm_provider) if model_info is None else model_info
    )
    priced: Final = NativePrices(row, custom_llm_provider).calculate(
        billed_quantities(usage),
        call_context(service_tier, data_residency, vertex_location, current_time),
    )
    return priced.prompt_cost, priced.completion_cost


def _coerce_token_count(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


@dataclass(frozen=True, slots=True)
class BilledTokenRates:
    """Per-token rates one request's usage bills at, after token tiers, off-peak windows and the
    regional multipliers the totals apply, so each cost line equals its token count times its rate."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float
    cache_read_input_audio_token_cost: float
    cache_creation_input_token_cost: float
    cache_creation_input_token_cost_by_ttl: Mapping[str, float]
    output_cost_per_reasoning_token: float
    cache_read_input_image_token_cost: float | None = None

    def scaled(self, multiplier: float) -> "BilledTokenRates":
        if multiplier == 1.0:
            return self
        return BilledTokenRates(
            input_cost_per_token=self.input_cost_per_token * multiplier,
            output_cost_per_token=self.output_cost_per_token * multiplier,
            cache_read_input_token_cost=self.cache_read_input_token_cost * multiplier,
            cache_read_input_audio_token_cost=self.cache_read_input_audio_token_cost * multiplier,
            cache_creation_input_token_cost=self.cache_creation_input_token_cost * multiplier,
            cache_creation_input_token_cost_by_ttl=MappingProxyType(
                {ttl: rate * multiplier for ttl, rate in self.cache_creation_input_token_cost_by_ttl.items()}
            ),
            output_cost_per_reasoning_token=self.output_cost_per_reasoning_token * multiplier,
            cache_read_input_image_token_cost=(
                self.cache_read_input_image_token_cost * multiplier
                if self.cache_read_input_image_token_cost is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TokenTypeCostBreakdown:
    reasoning_cost: float
    cache_read_cost: float
    cache_creation_cost: float
    rates: BilledTokenRates | None = None
    """Rates these lines were billed at, so a caller reporting both cannot resolve them a second,
    differently-argued way. None when the model's pricing could not be resolved."""


def _reasoning_token_count(usage: Usage) -> int:
    parsed: Final = (
        parse_completion_tokens_details(usage)["reasoning_tokens"] if usage.completion_tokens_details is not None else 0
    )
    return parsed or _coerce_token_count(getattr(usage, "reasoning_tokens", 0))


def _cache_token_counts(usage: Usage) -> tuple[int, int, int, CacheCreationTokenDetails | None]:
    """(cache read tokens, cached audio tokens, cache creation tokens, cache creation details): read from
    prompt_tokens_details first, then the private top-level counters the Usage constructor mirrors cache
    tokens onto for providers/callers that bypass the details."""
    parsed: Final = parse_prompt_tokens_details(usage) if usage.prompt_tokens_details is not None else None
    parsed_read: Final = parsed["cache_hit_tokens"] if parsed is not None else 0
    parsed_creation: Final = parsed["cache_creation_tokens"] if parsed is not None else 0
    return (
        parsed_read or _coerce_token_count(getattr(usage, "_cache_read_input_tokens", 0)),
        parsed["cache_hit_audio_tokens"] if parsed is not None else 0,
        parsed_creation or _coerce_token_count(getattr(usage, "_cache_creation_input_tokens", 0)),
        parsed["cache_creation_token_details"] if parsed is not None else None,
    )


def _custom_pricing_rates(custom_cost_per_token: CostPerToken) -> BilledTokenRates:
    """Flat custom pricing has no tiers, uplifts or reasoning rate: cache tokens bill at the configured
    cache rates (else the input rate) and reasoning at the output rate, as _cost_per_token_custom_pricing_helper does."""
    input_rate: Final = custom_cost_per_token["input_cost_per_token"]
    output_rate: Final = custom_cost_per_token["output_cost_per_token"]
    cache_read_rate: Final = custom_cost_per_token.get("cache_read_input_token_cost", input_rate)
    cache_creation_rate: Final = custom_cost_per_token.get("cache_creation_input_token_cost", input_rate)
    return BilledTokenRates(
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
        cache_read_input_token_cost=cache_read_rate,
        cache_read_input_audio_token_cost=cache_read_rate,
        cache_creation_input_token_cost=cache_creation_rate,
        cache_creation_input_token_cost_by_ttl=MappingProxyType({}),
        output_cost_per_reasoning_token=output_rate,
    )


def _cost_map_billed_rates(
    model_info: ModelInfo,
    usage: Usage,
    custom_llm_provider: str | None,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    current_time: datetime | None,
) -> BilledTokenRates:
    quote: Final = NativePrices(model_info, custom_llm_provider).quote(
        _count(usage.prompt_tokens),
        call_context(service_tier, data_residency, vertex_location, current_time),
    )
    cache_read_rate: Final = billed_cache_read_rate(quote, usage)
    multiplier: Final = (
        quote.uplift
        * get_provider_specific_geo_multiplier(model_info=model_info, usage=usage)
        * get_provider_specific_speed_multiplier(model_info=model_info, usage=usage)
    )
    return BilledTokenRates(
        input_cost_per_token=quote.input,
        output_cost_per_token=quote.output,
        cache_read_input_token_cost=cache_read_rate,
        cache_read_input_audio_token_cost=(
            cache_read_rate if quote.cache_read_audio is None else quote.cache_read_audio
        ),
        cache_creation_input_token_cost=quote.cache_write,
        cache_creation_input_token_cost_by_ttl=MappingProxyType(
            {"1hr": quote.cache_write_1h, "12hr": quote.cache_write_12h}
        ),
        output_cost_per_reasoning_token=quote.reasoning,
        cache_read_input_image_token_cost=quote.cache_read_image,
    ).scaled(multiplier)


def get_billed_token_rates(
    model: str,
    custom_llm_provider: str | None,
    usage: Usage,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    current_time: datetime | None = None,
    custom_cost_per_token: CostPerToken | None = None,
    model_info: ModelInfo | None = None,
) -> BilledTokenRates | None:
    """Rates the cost calculator bills ``usage`` at, resolved exactly as the totals and the token-type
    breakdown resolve them. None when the model's pricing cannot be resolved."""
    if custom_cost_per_token is not None:
        return _custom_pricing_rates(custom_cost_per_token)
    try:
        resolved_model_info: Final = (
            model_info
            if model_info is not None
            else get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        )
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model: no rates
        return None
    return _cost_map_billed_rates(
        model_info=resolved_model_info,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        current_time=current_time,
    )


def get_token_type_cost_breakdown(
    model: str,
    custom_llm_provider: str | None,
    usage: Usage,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    current_time: datetime | None = None,
    custom_cost_per_token: CostPerToken | None = None,
    model_info: ModelInfo | None = None,
) -> TokenTypeCostBreakdown:
    """
    Provider-agnostic cost of reasoning and cache tokens, derived from the usage
    object and model pricing alone.

    This works for every provider, including Perplexity/Cerebras/Dashscope whose
    cost calculators bypass ``generic_cost_per_token``, because cache tokens always
    land on ``prompt_tokens_details`` (via the Usage constructor and provider
    transformations) and reasoning tokens on ``completion_tokens_details``. It reuses
    the same rate resolution as the total-cost path (``get_billed_token_rates``) so the
    breakdown can never drift from the totals. A deployment billed by
    ``custom_cost_per_token`` is priced from those flat rates instead of the cost map and,
    like its totals, bills cache writes flat rather than by their 5m/1h split.
    Returns zeros (never raises) when the model or its pricing cannot be resolved.
    """
    rates: Final = get_billed_token_rates(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        current_time=current_time,
        custom_cost_per_token=custom_cost_per_token,
        model_info=model_info,
    )
    if rates is None:
        return TokenTypeCostBreakdown(0.0, 0.0, 0.0)

    cache_read_tokens, cached_audio_tokens, cache_creation_tokens, cache_creation_token_details = _cache_token_counts(
        usage
    )
    cached_image_tokens: Final = parse_prompt_tokens_details(usage)["cached_image_tokens"]
    image_cache_read_rate: Final = (
        rates.cache_read_input_image_token_cost
        if rates.cache_read_input_image_token_cost is not None
        else rates.cache_read_input_token_cost
    )
    cache_creation_cost: Final = (
        float(cache_creation_tokens) * rates.cache_creation_input_token_cost
        if custom_cost_per_token is not None
        else calculate_cache_writing_cost(
            cache_creation_tokens=cache_creation_tokens,
            cache_creation_token_details=cache_creation_token_details,
            cache_creation_cost=rates.cache_creation_input_token_cost,
            cache_creation_cost_by_ttl=rates.cache_creation_input_token_cost_by_ttl,
        )
    )
    return TokenTypeCostBreakdown(
        reasoning_cost=float(_reasoning_token_count(usage)) * rates.output_cost_per_reasoning_token,
        cache_read_cost=(
            float(cache_read_tokens - cached_audio_tokens - cached_image_tokens) * rates.cache_read_input_token_cost
            + float(cached_audio_tokens) * rates.cache_read_input_audio_token_cost
            + float(cached_image_tokens) * image_cache_read_rate
        ),
        cache_creation_cost=cache_creation_cost,
        rates=rates,
    )


def calculate_prompt_caching_savings(
    model_info: ModelInfo,
    usage: Usage,
    custom_llm_provider: str | None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    billed_at: datetime | None = None,
) -> float:
    """Read discount minus write premium, using the biller's rate and TTL resolution.

    Missing reads and unpublished (missing/zero) writes claim no saving or premium;
    explicit zero reads remain free. An unpublished 1h price uses the ordinary write rate.
    ``billed_at`` is the request's completion time, so off-peak windows resolve as the
    biller saw them rather than at the later spend write.
    """
    quote: Final = NativePrices(model_info, custom_llm_provider).quote(
        _count(usage.prompt_tokens),
        call_context(service_tier, data_residency, vertex_location, billed_at),
    )
    prompt_base_cost: Final = quote.input
    effective_cache_read_cost: Final = billed_cache_read_rate(quote, usage)
    write_rate: Final = quote.cache_write or prompt_base_cost
    prompt_tokens_details: Final = parse_prompt_tokens_details(usage)
    cache_read_tokens: Final = max(prompt_tokens_details["cache_hit_tokens"], 0)
    cache_creation_tokens: Final = max(prompt_tokens_details["cache_creation_tokens"], 0)
    details: Final = prompt_tokens_details["cache_creation_token_details"]
    cache_creation_details: Final = (
        CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=max(details.ephemeral_5m_input_tokens or 0, 0),
            ephemeral_30m_input_tokens=max(details.ephemeral_30m_input_tokens or 0, 0),
            ephemeral_1h_input_tokens=max(details.ephemeral_1h_input_tokens or 0, 0),
            ephemeral_12h_input_tokens=max(details.ephemeral_12h_input_tokens or 0, 0),
        )
        if details is not None
        else None
    )
    read_discount: Final = cache_read_tokens * max(prompt_base_cost - effective_cache_read_cost, 0.0)
    write_premium: Final = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_details,
        cache_creation_cost=write_rate - prompt_base_cost,
        cache_creation_cost_by_ttl=MappingProxyType(
            {"1hr": quote.cache_write_1h - prompt_base_cost, "12hr": quote.cache_write_12h - prompt_base_cost}
        ),
    )
    return (read_discount - write_premium) * quote.uplift


def calculate_image_response_cost_from_usage(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: str,
    model_info: ModelInfo | None = None,
    vertex_location: str | None = None,
    service_tier: str | None = None,
) -> float | None:
    """
    Calculate image generation cost from usage metadata when available.

    Returns:
        Optional[float]: total cost from token usage, or None when usage metadata
        is missing/incomplete and caller should fall back to flat per-image pricing.
    """
    usage: Final = image_response.usage
    if usage is None:
        return None

    prompt_tokens: Final = usage.input_tokens
    completion_tokens: Final = usage.output_tokens
    total_tokens: Final = usage.total_tokens

    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None

    # ImageResponse may carry a default zeroed usage object even when provider
    # usage metadata is absent. Treat this as missing usage and fall back.
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    if model_info is not None and not prices_tokens(model_info):
        return None

    input_tokens_details: Final[object] = getattr(usage, "input_tokens_details", None)
    prompt_tokens_details: PromptTokensDetailsWrapper | None = None
    if input_tokens_details is not None:
        # input_tokens_details may be a dict (e.g. OpenAI image edit responses)
        # or an object; read it tolerantly like the output side below, so image
        # input tokens are priced at input_cost_per_image_token instead of
        # silently falling back to the text rate.
        prompt_tokens_details = PromptTokensDetailsWrapper(
            text_tokens=_get_token_detail_value(input_tokens_details, "text_tokens"),
            image_tokens=_get_token_detail_value(input_tokens_details, "image_tokens"),
            cached_tokens=0,
        )

    completion_tokens_details_attr: Final[object] = getattr(usage, "completion_tokens_details", None)
    output_tokens_details: Final[object] = (
        getattr(usage, "output_tokens_details", None)
        if completion_tokens_details_attr is None
        else completion_tokens_details_attr
    )

    text_tokens: Final = _get_token_detail_value(output_tokens_details, "text_tokens") or 0
    image_tokens: Final = _get_token_detail_value(output_tokens_details, "image_tokens") or 0
    audio_tokens: Final = _get_token_detail_value(output_tokens_details, "audio_tokens") or 0
    reasoning_tokens: Final = _get_token_detail_value(output_tokens_details, "reasoning_tokens") or 0
    has_output_modalities: Final = any(
        _get_token_detail_value(output_tokens_details, modality) is not None
        for modality in ("text_tokens", "image_tokens", "audio_tokens")
    )
    unitemized_tokens: Final = max(completion_tokens - text_tokens - image_tokens - audio_tokens - reasoning_tokens, 0)
    unitemized_image_tokens: Final = unitemized_tokens if not has_output_modalities and image_response.data else 0
    completion_tokens_details: Final = CompletionTokensDetailsWrapper(
        text_tokens=text_tokens + unitemized_tokens - unitemized_image_tokens,
        image_tokens=image_tokens + unitemized_image_tokens,
        reasoning_tokens=reasoning_tokens,
        audio_tokens=audio_tokens,
    )

    normalized_usage: Final = Usage(
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
        model_info=model_info,
        vertex_location=vertex_location,
        service_tier=service_tier,
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
    usage: Final = image_response.usage
    if usage is None:
        return 0.0

    web_search_requests: Final = getattr(usage, "web_search_requests", None)
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


class _ModelPricedImageCostCalculator(Protocol):
    def __call__(self, model: str, image_response: ImageResponse, model_info: ModelInfo | None = None) -> float: ...


def _model_priced_image_cost_calculator(
    custom_llm_provider: str | None, model: str, pricing: ModelInfo | None
) -> _ModelPricedImageCostCalculator | None:
    match custom_llm_provider:
        case litellm.LlmProviders.RECRAFT.value:
            from litellm.llms.recraft.cost_calculator import cost_calculator as recraft_image_cost_calculator

            return recraft_image_cost_calculator
        case litellm.LlmProviders.AIML.value:
            from litellm.llms.aiml.image_generation.cost_calculator import cost_calculator as aiml_image_cost_calculator

            return aiml_image_cost_calculator
        case litellm.LlmProviders.COMETAPI.value:
            from litellm.llms.cometapi.image_generation.cost_calculator import (
                cost_calculator as cometapi_image_cost_calculator,
            )

            return cometapi_image_cost_calculator
        case litellm.LlmProviders.RUNWAYML.value:
            from litellm.llms.runwayml.cost_calculator import cost_calculator as runwayml_image_cost_calculator

            return runwayml_image_cost_calculator
        case litellm.LlmProviders.TENCENT.value:
            from litellm.llms.tencent.cost_calculator import image_cost_calculator as tencent_image_cost_calculator
            from litellm.llms.tencent.image_generation.transformation import is_tokenhub_image_model

            return tencent_image_cost_calculator if is_tokenhub_image_model(model) else None
        case litellm.LlmProviders.DASHSCOPE.value:
            if _deployment_prices_images_without_an_output_price(pricing):
                return None
            from litellm.llms.dashscope.image_generation.cost_calculator import (
                cost_calculator as dashscope_image_cost_calculator,
            )

            return dashscope_image_cost_calculator
        case _:
            return None


class CostCalculatorUtils:
    @staticmethod
    def call_type_has_image_response(call_type: str) -> bool:
        """
        Returns True if the call type has an image response

        eg calls that have image response:
        - Image Generation
        - Image Edit
        - Passthrough Image Generation
        """
        return call_type in _IMAGE_RESPONSE_CALL_TYPES

    _call_type_has_image_response = call_type_has_image_response

    @staticmethod
    def route_image_generation_cost_calculator(
        model: str,
        completion_response: ImageResponse,
        custom_llm_provider: str | None = None,
        quality: str | None = None,
        n: int | None = None,
        size: str | None = None,
        optional_params: dict | None = None,
        call_type: str | None = None,
        model_info: ModelInfo | None = None,
        vertex_location: str | None = None,
        service_tier: str | None = None,
    ) -> float:
        """
        Route the image generation cost calculator based on the custom_llm_provider

        ``model_info`` is the deployment's own price table. Its valid prices are laid over the shared
        cost-map entry and handed to the provider calculator, so per-image, per-pixel and per-token
        deployment prices all apply while provider logic (token-first billing, grounding surcharges,
        image counting) stays in one place. An unparseable price is logged and ignored.
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
        from litellm.llms.vertex_ai.image_generation.cost_calculator import (
            cost_calculator as vertex_ai_image_cost_calculator,
        )

        resolved_size: Final = (
            size or completion_response.size or _requested_image_size(optional_params) or "1024-x-1024"
        )
        resolved_quality: Final = (
            quality or completion_response.quality or _requested_image_param(optional_params, "quality") or "standard"
        )
        resolved_n: Final = n if n is not None else (len(completion_response.data) if completion_response.data else 0)
        pricing: Final = deployment_pricing(model_info)
        if custom_llm_provider == litellm.LlmProviders.XAI.value:
            from litellm.llms.xai.image_generation.cost_calculator import cost_calculator as xai_image_cost_calculator

            xai_cost: Final = xai_image_cost_calculator(
                model=model,
                image_response=completion_response,
                quality=quality,
                n=resolved_n,
                optional_params=_IMAGE_REQUEST_PARAMS.validate_python(optional_params),
                call_type=call_type,
                model_info=pricing,
            )
            if xai_cost is not None:
                return xai_cost
        model_priced_calculator: Final = _model_priced_image_cost_calculator(custom_llm_provider, model, pricing)

        if model_priced_calculator is not None:
            return model_priced_calculator(model=model, image_response=completion_response, model_info=pricing)
        if custom_llm_provider == litellm.LlmProviders.VERTEX_AI.value:
            if isinstance(completion_response, ImageResponse):
                return vertex_ai_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                    model_info=pricing,
                    vertex_location=vertex_location,
                    service_tier=service_tier,
                )
        elif custom_llm_provider == litellm.LlmProviders.BEDROCK.value:
            if isinstance(completion_response, ImageResponse):
                return bedrock_image_cost_calculator(
                    model=model,
                    size=resolved_size,
                    image_response=completion_response,
                    optional_params=optional_params,
                )
            raise TypeError("completion_response must be of type ImageResponse for bedrock image cost calculation")
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
                    model_info=pricing,
                )
            from litellm.llms.gemini.image_generation.cost_calculator import (
                cost_calculator as gemini_image_cost_calculator,
            )

            return gemini_image_cost_calculator(
                model=model,
                image_response=completion_response,
                model_info=pricing,
            )
        elif custom_llm_provider == litellm.LlmProviders.AZURE_AI.value:
            return azure_ai_image_cost_calculator(
                model=model,
                image_response=completion_response,
                size=resolved_size,
                n=resolved_n,
                optional_params=optional_params,
                model_info=pricing,
            )
        elif custom_llm_provider == litellm.LlmProviders.FAL_AI.value:
            from litellm.llms.fal_ai.cost_calculator import (
                cost_calculator as fal_ai_image_cost_calculator,
            )

            return fal_ai_image_cost_calculator(
                model=model,
                image_response=completion_response,
                optional_params=optional_params,
                model_info=pricing,
            )
        elif (
            custom_llm_provider == litellm.LlmProviders.OPENAI.value
            or custom_llm_provider == litellm.LlmProviders.AZURE.value
        ):
            # gpt-image models use token-based pricing.
            model_lower: Final = model.lower()
            if "gpt-image" in model_lower:
                from litellm.llms.openai.image_generation.cost_calculator import (
                    cost_calculator as openai_gpt_image_cost_calculator,
                )

                return openai_gpt_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                    custom_llm_provider=custom_llm_provider,
                    model_info=pricing,
                )
            # Fall through to default for DALL-E models
            return default_image_cost_calculator(
                model=model,
                quality=resolved_quality,
                custom_llm_provider=custom_llm_provider,
                n=resolved_n,
                size=resolved_size,
                optional_params=optional_params,
                model_info=pricing,
            )
        else:
            return default_image_cost_calculator(
                model=model,
                quality=resolved_quality,
                custom_llm_provider=custom_llm_provider,
                n=resolved_n,
                size=resolved_size,
                optional_params=optional_params,
                model_info=pricing,
            )
        return 0.0
