from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import (
    Final,
    Literal,
    cast,  # noqa: TID251  # validated request prices extend the read-only public ModelInfo shape
)

from pydantic import TypeAdapter

from litellm.exceptions import ModelNotMappedError
from litellm.rust_bridge import _native
from litellm.types.utils import CacheCreationTokenDetails, CustomPricingLiteLLMParams, ModelInfo, Usage

_REQUEST_ROW: Final = TypeAdapter(dict[str, object])
_REQUEST_PRICE_KEYS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)


def _request_metadata(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    info: Final = kwargs.get("model_info")
    if not isinstance(info, Mapping):
        return {}
    fields: Final = _REQUEST_ROW.validate_python(info)
    return {
        key: value
        for key, value in fields.items()
        if key in ("mode", "supports_prompt_caching", "max_tokens") and value is not None
    }


def request_prices(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    has_tokens: Final = (
        kwargs.get("input_cost_per_token") is not None and kwargs.get("output_cost_per_token") is not None
    )
    has_seconds: Final = any(
        kwargs.get(key) is not None for key in ("cost_per_second", "input_cost_per_second", "output_cost_per_second")
    )
    if not has_tokens and not has_seconds:
        return None
    prices: Final = CustomPricingLiteLLMParams.model_validate(
        {key: value for key, value in kwargs.items() if key in _REQUEST_PRICE_KEYS and value is not None}
    )
    return {
        **_request_metadata(kwargs),
        **_REQUEST_ROW.validate_python(prices.model_dump(exclude_none=True, exclude_unset=True)),
    }


def request_model_info(model: str, provider: str | None, overrides: Mapping[str, object]) -> ModelInfo:
    from litellm.utils import get_model_info

    def live_info() -> ModelInfo:
        try:
            return get_model_info(model=model, custom_llm_provider=provider)
        except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for an unknown custom model
            return {
                "key": model,
                "litellm_provider": provider or "custom",
                "mode": "chat",
                "input_cost_per_token": None,
                "output_cost_per_token": None,
                "max_tokens": None,
                "max_input_tokens": None,
                "max_output_tokens": None,
                "supported_openai_params": None,
            }

    return cast(  # cast-ok: validated pricing keys preserve extra request meters
        ModelInfo, {**live_info(), **overrides}
    )


def price(
    row: Mapping[str, object],
    usage: _native.UsageInput,
    when: _native.WhenInput,
    *,
    factors: Sequence[float] = (),
    cache_read_default_ratio: float | None = None,
    directional_token_threshold: int | None = None,
) -> _native.Priced | _native.Unpriced | _native.RowErrors:
    if not factors:
        return _native.price(
            dict(row),
            usage,
            when,
            policy="sdk_tokens",
            cache_read_default_ratio=cache_read_default_ratio,
            directional_token_threshold=directional_token_threshold,
        )
    result: Final = evaluate(
        row,
        usage,
        when,
        factors=factors,
        cache_read_default_ratio=cache_read_default_ratio,
        directional_token_threshold=directional_token_threshold,
    )
    return result if isinstance(result, _native.RowErrors) else result.pricing


def evaluate(
    row: Mapping[str, object],
    usage: _native.UsageInput,
    when: _native.WhenInput,
    *,
    project: Sequence[str] = (),
    factors: Sequence[float] = (),
    cache_read_default_ratio: float | None = None,
    directional_token_threshold: int | None = None,
) -> _native.PriceEvaluation | _native.RowErrors:
    return _native.price_with_rates(
        dict(row),
        usage,
        when,
        policy="sdk_tokens",
        project=project,
        factors=factors,
        cache_read_default_ratio=cache_read_default_ratio,
        directional_token_threshold=directional_token_threshold,
    )


def require_priced(result: _native.Priced | _native.Unpriced | _native.RowErrors, model: str) -> _native.Priced:
    if isinstance(result, _native.Priced):
        return result
    detail: Final = (
        result.missing
        if isinstance(result, _native.Unpriced)
        else tuple(f"{error.field}: {error.message}" for error in result.errors)
    )
    raise ModelNotMappedError(f"No valid pricing for model={model}: {detail}")


def price_quantities(
    row: Mapping[str, object],
    quantities: Mapping[str, int],
    *,
    model: str = "",
    factors: Sequence[float] = (),
) -> _native.Priced:
    from litellm._internal_context import current_billing_time

    return require_priced(
        price(
            {"litellm_provider": "custom", **row},
            _native.UsageInput(dict(quantities)),
            billing_context(current_billing_time()),
            factors=factors,
        ),
        model,
    )


def billing_context(
    current_time: datetime,
    service_tier: str | None = None,
    channel: Literal["online", "batch", "decisions"] = "online",
) -> _native.WhenInput:
    normalized: Final = service_tier.strip().lower() if service_tier else "standard"
    tier: Final = (
        "priority"
        if normalized == "fast"
        else normalized
        if normalized in ("priority", "flex", "ultrafast", "balanced")
        else "standard"
    )
    utc_time: Final = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=timezone.utc)
    return _native.WhenInput(int(utc_time.timestamp()), tier=tier, channel=channel)


def cache_read_meter(usage: Usage) -> Literal["cache_read_input_token_cost", "explicit_cache_read_input_token_cost"]:
    return (
        "explicit_cache_read_input_token_cost"
        if getattr(usage.prompt_tokens_details, "cache_type", None) == "ephemeral"
        else "cache_read_input_token_cost"
    )


def cache_write_quantities(cache_creation: int, ttl: CacheCreationTokenDetails | None) -> Mapping[str, int]:
    short_writes: Final = ((ttl.ephemeral_5m_input_tokens or 0) + (ttl.ephemeral_30m_input_tokens or 0)) if ttl else 0
    writes: Final = {
        "cache_creation_input_token_cost_above_1hr": (ttl.ephemeral_1h_input_tokens or 0) if ttl else 0,
        "cache_creation_input_token_cost_above_12hr": (ttl.ephemeral_12h_input_tokens or 0) if ttl else 0,
    }
    return {
        "cache_creation_input_token_cost": max(cache_creation - short_writes - sum(writes.values()), 0) + short_writes,
        **writes,
    }


def _prompt_quantities(usage: Usage, row: Mapping[str, object]) -> Mapping[str, int]:
    from litellm.litellm_core_utils.llm_cost_calc.utils import parse_prompt_tokens_details

    if usage.prompt_tokens_details is None:
        return {"input_cost_per_token": usage.prompt_tokens}
    details: Final = parse_prompt_tokens_details(usage)
    cache_hit: Final = details["cache_hit_tokens"]
    cache_creation: Final = details["cache_creation_tokens"]
    total_details: Final = sum(
        (
            details["text_tokens"],
            cache_hit,
            cache_creation,
            details["audio_tokens"],
            details["image_tokens"],
            details["video_tokens"],
        )
    )
    overlaps: Final = (cache_hit > 0 or cache_creation > 0) and total_details > usage.prompt_tokens
    uncached: Final = max(usage.prompt_tokens - cache_hit - cache_creation, 0)
    audio: Final = min(details["audio_tokens"], uncached) if overlaps else details["audio_tokens"]
    image: Final = min(details["image_tokens"], uncached - audio) if overlaps else details["image_tokens"]
    video: Final = min(details["video_tokens"], uncached - audio - image) if overlaps else details["video_tokens"]
    remainder: Final = max(uncached - audio - image - video, 0)
    text: Final = (
        remainder
        if overlaps or (details["text_tokens"] == 0 and details["image_count"] == 0)
        else details["text_tokens"]
    )
    return {
        "input_cost_per_token": 0
        if details["character_count"] and row.get("input_cost_per_character") is not None
        else text,
        "input_cost_per_audio_token": 0
        if details["audio_length_seconds"] and row.get("input_cost_per_audio_per_second") is not None
        else audio,
        "input_cost_per_image_token": 0
        if details["image_count"] and row.get("input_cost_per_image") is not None
        else image,
        "input_cost_per_video_token": 0
        if details["video_length_seconds"] and row.get("input_cost_per_video_per_second") is not None
        else video,
        cache_read_meter(usage): cache_hit - details["cache_hit_audio_tokens"],
        "cache_read_input_audio_token_cost": details["cache_hit_audio_tokens"],
        **cache_write_quantities(cache_creation, details["cache_creation_token_details"]),
    }


def _completion_quantities(usage: Usage, *, batch: bool = False) -> Mapping[str, int]:
    from litellm.litellm_core_utils.llm_cost_calc.utils import parse_completion_tokens_details

    if usage.completion_tokens_details is None:
        return {"output_cost_per_token": usage.completion_tokens}
    details: Final = parse_completion_tokens_details(usage)
    image_tokens: Final = min(details["image_tokens"], usage.completion_tokens) if batch else details["image_tokens"]
    modalities: Final = {
        "output_cost_per_audio_token": details["audio_tokens"],
        "output_cost_per_reasoning_token": details["reasoning_tokens"],
        "output_cost_per_image_token": image_tokens,
        "output_cost_per_video_token": details["video_tokens"],
    }
    text: Final = details["text_tokens"] or max(usage.completion_tokens - sum(modalities.values()), 0)
    return {"output_cost_per_token": text, **modalities}


def token_usage(
    usage: Usage,
    row: Mapping[str, object],
    *,
    inclusive_threshold: bool = False,
    batch: bool = False,
    direction: Literal["input", "output"] | None = None,
) -> _native.UsageInput:
    return _native.UsageInput(
        {
            **(_prompt_quantities(usage, row) if direction != "output" else {}),
            **(_completion_quantities(usage, batch=batch) if direction != "input" else {}),
        },
        prompt_tokens=usage.prompt_tokens,
        inclusive_threshold=inclusive_threshold,
    )
