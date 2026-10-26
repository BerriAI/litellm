# What is this?
## Helper utilities for cost_per_token()

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import prod
from types import MappingProxyType
from typing import Final, Literal, TypedDict, cast

from typing_extensions import ReadOnly

import litellm
from litellm._internal_context import current_billing_time
from litellm._logging import verbose_logger
from litellm.constants import FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO
from litellm.rust_bridge import _native
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CallTypes,
    CompletionTokensDetailsWrapper,
    CostPerToken,
    CustomPricingLiteLLMParams,
    DataResidency,
    ImageResponse,
    ModelInfo,
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

# Pre-resolved DataResidency enum values for fast membership checks
_VALID_DATA_RESIDENCIES: Final = frozenset(r.value for r in DataResidency)

_DEPLOYMENT_PRICING_KEYS: Final[frozenset[str]] = frozenset(CustomPricingLiteLLMParams.model_fields)

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


def uses_inclusive_token_thresholds(custom_llm_provider: str | None) -> bool:
    return custom_llm_provider in _INCLUSIVE_THRESHOLD_PROVIDERS


def cache_read_default_ratio(custom_llm_provider: str | None) -> float | None:
    return FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO if custom_llm_provider == "fireworks_ai" else None


def _get_token_detail_value(details: object, key: str) -> int | None:
    if isinstance(details, dict):
        value = details.get(key)
    else:
        value = getattr(details, key, None)
    return value if isinstance(value, int) else None


_IMAGE_SIZE_PATTERN: Final = re.compile(r"\d+(?:x|-x-)\d+")


def _requested_image_param(optional_params: Mapping[str, object] | None, key: str) -> str | None:
    value: Final = None if optional_params is None else optional_params.get(key)
    return value if isinstance(value, str) else None


def _requested_image_size(optional_params: Mapping[str, object] | None) -> str | None:
    value: Final = _requested_image_param(optional_params, "size")
    return value if value is not None and _IMAGE_SIZE_PATTERN.fullmatch(value) else None


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


def calculate_cost_component(model_info: ModelInfo, cost_key: str, usage_value: float | None) -> float:
    """
    Generic cost calculator for any usage component

    Args:
        model_info: Dictionary containing cost information
        cost_key: The key for the cost multiplier in model_info (e.g., 'input_cost_per_audio_token')
        usage_value: The actual usage value (e.g., number of tokens, characters, seconds)

    Returns:
        float: The calculated cost
    """
    cost_per_unit: Final = get_cost_per_unit(model_info, cost_key)
    if cost_per_unit is not None and isinstance(cost_per_unit, float) and usage_value is not None and usage_value > 0:
        return float(usage_value) * cost_per_unit
    return 0.0


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


class PromptTokensDetailsResult(TypedDict):
    cache_hit_tokens: int
    cache_hit_audio_tokens: ReadOnly[int]
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
    multiplier: Final = model_info.get(f"regional_processing_uplift_multiplier_{residency}")
    if multiplier is None:
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


def get_vertex_regional_endpoint_uplift(model_info: ModelInfo, vertex_location: str | None) -> float:
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
        - data_residency: optional OpenAI data-residency region (e.g. "eu", "us"),
          used to apply the per-model regional-processing uplift multiplier.
        - vertex_location: optional Vertex AI location the request was served from
          (e.g. "us-east5", "global"), used to apply the per-model
          regional-endpoint uplift multiplier when non-global.
        - current_time: the moment the request is billed at, for off_peak_pricing; defaults to now, UTC

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """

    ## GET MODEL INFO
    # A caller that already resolved the deployment's effective rates passes them in
    # rather than handing back a name for this to re-resolve. A name cannot express a
    # per-deployment override: those are registered under the deployment id and kept off
    # the shared model-name key, so resolving from the name here reads the public rate.
    resolved_model_info: Final = (
        get_model_info(model=model, custom_llm_provider=custom_llm_provider) if model_info is None else model_info
    )

    factors: Final = token_pricing_factors(
        resolved_model_info, usage, custom_llm_provider, data_residency, vertex_location
    )
    return price_token_usage(
        model, usage, custom_llm_provider, resolved_model_info, service_tier, current_time, factors
    )


def price_token_usage(
    model: str,
    usage: Usage,
    custom_llm_provider: str,
    resolved_model_info: ModelInfo,
    service_tier: str | None,
    current_time: datetime | None,
    factors: Sequence[float],
) -> tuple[float, float]:
    from litellm._cost import billing_context, price, require_priced, token_usage

    billing_time: Final = current_time if current_time is not None else current_billing_time()
    priced: Final = require_priced(
        price(
            {"litellm_provider": custom_llm_provider, **resolved_model_info},
            token_usage(
                usage,
                resolved_model_info,
                inclusive_threshold=uses_inclusive_token_thresholds(custom_llm_provider),
            ),
            billing_context(billing_time, service_tier),
            factors=factors,
            cache_read_default_ratio=cache_read_default_ratio(custom_llm_provider),
        ),
        model,
    )
    details: Final = parse_prompt_tokens_details(usage)
    non_token_input: Final = sum(
        calculate_cost_component(resolved_model_info, key, quantity)
        for key, quantity in (
            ("input_cost_per_character", details["character_count"]),
            ("input_cost_per_image", details["image_count"]),
            ("input_cost_per_video_per_second", details["video_length_seconds"]),
            ("input_cost_per_audio_per_second", details["audio_length_seconds"]),
            ("input_cost_per_query", details["query_count"]),
        )
    )
    return priced.input + non_token_input * prod(factors), priced.output


def token_pricing_factors(
    model_info: ModelInfo,
    usage: Usage,
    custom_llm_provider: str | None,
    data_residency: str | None,
    vertex_location: str | None,
) -> tuple[float, ...]:
    regional: Final = (
        get_regional_uplift_multiplier(model_info, data_residency),
        get_vertex_regional_endpoint_uplift(model_info, vertex_location),
    )
    if custom_llm_provider != "anthropic":
        return regional
    return (
        *regional,
        get_provider_specific_speed_multiplier(model_info, usage),
        get_provider_specific_geo_multiplier(model_info, usage),
    )


def _coerce_token_count(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


@dataclass(frozen=True, slots=True)
class BilledTokenRates:
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    cache_read_input_token_cost: float | None
    cache_read_input_audio_token_cost: float | None
    cache_creation_input_token_cost: float | None
    cache_creation_input_token_cost_by_ttl: Mapping[str, float | None]
    output_cost_per_reasoning_token: float | None


@dataclass(frozen=True, slots=True)
class TokenTypeCostBreakdown:
    reasoning_cost: float
    cache_read_cost: float
    cache_creation_cost: float
    rates: BilledTokenRates | None = None
    """Rates these lines were billed at, so a caller reporting both cannot resolve them a second,
    differently-argued way. None when the model's pricing could not be resolved."""


_BILLED_TOKEN_METERS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "explicit_cache_read_input_token_cost",
    "cache_read_input_audio_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_12hr",
    "output_cost_per_reasoning_token",
)


def _reported_token_evaluation(
    model: str,
    custom_llm_provider: str | None,
    usage: Usage,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    current_time: datetime | None,
    custom_cost_per_token: CostPerToken | None,
    model_info: ModelInfo | None,
) -> _native.PriceEvaluation | _native.RowErrors | None:
    from litellm._cost import billing_context, evaluate, token_usage

    if custom_cost_per_token is not None:
        custom_row: Final = {"litellm_provider": "custom", **custom_cost_per_token}
        return evaluate(
            custom_row,
            token_usage(usage, custom_row),
            billing_context(current_time if current_time is not None else current_billing_time()),
            project=_BILLED_TOKEN_METERS,
        )
    try:
        resolved_model_info: Final = (
            model_info
            if model_info is not None
            else get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        )
    except Exception:  # noqa: BLE001  # preserve the public unknown-model reporting contract
        return None
    pricing_row: Final[Mapping[str, object]] = (
        resolved_model_info
        if "litellm_provider" in resolved_model_info
        else dict(resolved_model_info, litellm_provider=custom_llm_provider or "custom")
    )
    return evaluate(
        pricing_row,
        token_usage(usage, pricing_row, inclusive_threshold=uses_inclusive_token_thresholds(custom_llm_provider)),
        billing_context(current_time if current_time is not None else current_billing_time(), service_tier),
        project=_BILLED_TOKEN_METERS,
        factors=token_pricing_factors(resolved_model_info, usage, custom_llm_provider, data_residency, vertex_location),
        cache_read_default_ratio=cache_read_default_ratio(custom_llm_provider),
    )


def _billed_rates(evaluation: _native.PriceEvaluation, usage: Usage, *, flat_custom: bool = False) -> BilledTokenRates:
    from litellm._cost import cache_read_meter

    rates: Final = {rate.requested_meter: rate.unit_price for rate in evaluation.rates}
    return BilledTokenRates(
        input_cost_per_token=rates["input_cost_per_token"],
        output_cost_per_token=rates["output_cost_per_token"],
        cache_read_input_token_cost=rates[cache_read_meter(usage)],
        cache_read_input_audio_token_cost=rates["cache_read_input_audio_token_cost"],
        cache_creation_input_token_cost=rates["cache_creation_input_token_cost"],
        cache_creation_input_token_cost_by_ttl=MappingProxyType(
            {}
            if flat_custom
            else {
                "1hr": rates["cache_creation_input_token_cost_above_1hr"],
                "12hr": rates["cache_creation_input_token_cost_above_12hr"],
            }
        ),
        output_cost_per_reasoning_token=rates["output_cost_per_reasoning_token"],
    )


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
    evaluation: Final = _reported_token_evaluation(
        model,
        custom_llm_provider,
        usage,
        service_tier,
        data_residency,
        vertex_location,
        current_time,
        custom_cost_per_token,
        model_info,
    )
    if not isinstance(evaluation, _native.PriceEvaluation):
        return None
    return _billed_rates(evaluation, usage, flat_custom=custom_cost_per_token is not None)


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
    evaluation: Final = _reported_token_evaluation(
        model,
        custom_llm_provider,
        usage,
        service_tier,
        data_residency,
        vertex_location,
        current_time,
        custom_cost_per_token,
        model_info,
    )
    if not isinstance(evaluation, _native.PriceEvaluation) or not isinstance(evaluation.pricing, _native.Priced):
        return TokenTypeCostBreakdown(0.0, 0.0, 0.0)
    lines: Final = evaluation.pricing.lines
    return TokenTypeCostBreakdown(
        reasoning_cost=sum(
            line.amount for line in lines if (line.fallback_from or line.meter) == "output_cost_per_reasoning_token"
        ),
        cache_read_cost=sum(
            line.amount
            for line in lines
            if (line.fallback_from or line.meter)
            in (
                "cache_read_input_token_cost",
                "explicit_cache_read_input_token_cost",
                "cache_read_input_audio_token_cost",
            )
        ),
        cache_creation_cost=sum(
            line.amount
            for line in lines
            if (line.fallback_from or line.meter)
            in (
                "cache_creation_input_token_cost",
                "cache_creation_input_token_cost_above_1hr",
                "cache_creation_input_token_cost_above_12hr",
            )
        ),
        rates=_billed_rates(evaluation, usage, flat_custom=custom_cost_per_token is not None),
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
    from litellm._cost import billing_context, cache_read_meter, cache_write_quantities, price

    row: Final = {"litellm_provider": custom_llm_provider or "custom", **model_info}
    when: Final = billing_context(billed_at if billed_at is not None else current_billing_time(), service_tier)
    factors: Final = (
        get_regional_uplift_multiplier(model_info, data_residency),
        get_vertex_regional_endpoint_uplift(model_info, vertex_location),
    )

    def cache_cost(quantities: Mapping[str, int]) -> _native.Priced | None:
        result: Final = price(
            row,
            _native.UsageInput(
                dict(quantities),
                prompt_tokens=usage.prompt_tokens,
                inclusive_threshold=uses_inclusive_token_thresholds(custom_llm_provider),
            ),
            when,
            factors=factors,
            cache_read_default_ratio=cache_read_default_ratio(custom_llm_provider),
        )
        return result if isinstance(result, _native.Priced) else None

    details: Final = parse_prompt_tokens_details(usage)
    reads: Final = max(details["cache_hit_tokens"], 0)
    ttl: Final = details["cache_creation_token_details"]
    writes: Final = cache_write_quantities(
        max(details["cache_creation_tokens"], 0),
        CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=max(ttl.ephemeral_5m_input_tokens or 0, 0),
            ephemeral_30m_input_tokens=max(ttl.ephemeral_30m_input_tokens or 0, 0),
            ephemeral_1h_input_tokens=max(ttl.ephemeral_1h_input_tokens or 0, 0),
            ephemeral_12h_input_tokens=max(ttl.ephemeral_12h_input_tokens or 0, 0),
        )
        if ttl is not None
        else None,
    )
    read_meter: Final = cache_read_meter(usage)
    actual: Final = cache_cost({read_meter: reads, **writes})
    if actual is None:
        return 0.0
    priced_writes: Final = tuple(
        line
        for line in actual.lines
        if (line.fallback_from or line.meter) in writes
        and ((line.fallback_from or line.meter) != "cache_creation_input_token_cost" or line.unit_price > 0)
    )
    read_baseline: Final = cache_cost({"input_cost_per_token": reads})
    write_baseline: Final = cache_cost({"input_cost_per_token": sum(line.quantity for line in priced_writes)})
    if read_baseline is None or write_baseline is None:
        return 0.0
    read_cost: Final = sum(line.amount for line in actual.lines if (line.fallback_from or line.meter) == read_meter)
    write_cost: Final = sum(line.amount for line in priced_writes)
    return max(read_baseline.input - read_cost, 0.0) - (write_cost - write_baseline.input)


def calculate_image_response_cost_from_usage(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: str,
    model_info: ModelInfo | None = None,
    vertex_location: str | None = None,
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

    if output_tokens_details is None:
        completion_tokens_details = CompletionTokensDetailsWrapper(
            text_tokens=0,
            image_tokens=completion_tokens,
            reasoning_tokens=0,
            audio_tokens=0,
        )
    else:
        text_tokens = _get_token_detail_value(output_tokens_details, "text_tokens") or 0
        image_tokens: Final = _get_token_detail_value(output_tokens_details, "image_tokens") or 0
        audio_tokens: Final = _get_token_detail_value(output_tokens_details, "audio_tokens") or 0
        reasoning_tokens: Final = _get_token_detail_value(output_tokens_details, "reasoning_tokens") or 0
        known_output_tokens: Final = text_tokens + image_tokens + audio_tokens + reasoning_tokens
        if completion_tokens > known_output_tokens:
            text_tokens += completion_tokens - known_output_tokens

        completion_tokens_details = CompletionTokensDetailsWrapper(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
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
        from litellm.llms.recraft.cost_calculator import (
            cost_calculator as recraft_image_cost_calculator,
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

        if custom_llm_provider == litellm.LlmProviders.VERTEX_AI.value:
            if isinstance(completion_response, ImageResponse):
                return vertex_ai_image_cost_calculator(
                    model=model,
                    image_response=completion_response,
                    model_info=pricing,
                    vertex_location=vertex_location,
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
        elif custom_llm_provider == litellm.LlmProviders.RECRAFT.value:
            from litellm.llms.recraft.cost_calculator import (
                cost_calculator as recraft_image_cost_calculator,
            )

            return recraft_image_cost_calculator(
                model=model,
                image_response=completion_response,
                model_info=pricing,
            )
        elif custom_llm_provider == litellm.LlmProviders.AIML.value:
            from litellm.llms.aiml.image_generation.cost_calculator import (
                cost_calculator as aiml_image_cost_calculator,
            )

            return aiml_image_cost_calculator(
                model=model,
                image_response=completion_response,
                model_info=pricing,
            )
        elif custom_llm_provider == litellm.LlmProviders.COMETAPI.value:
            from litellm.llms.cometapi.image_generation.cost_calculator import (
                cost_calculator as cometapi_image_cost_calculator,
            )

            return cometapi_image_cost_calculator(
                model=model,
                image_response=completion_response,
                model_info=pricing,
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
        elif custom_llm_provider == litellm.LlmProviders.RUNWAYML.value:
            from litellm.llms.runwayml.cost_calculator import (
                cost_calculator as runwayml_image_cost_calculator,
            )

            return runwayml_image_cost_calculator(
                model=model,
                image_response=completion_response,
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
