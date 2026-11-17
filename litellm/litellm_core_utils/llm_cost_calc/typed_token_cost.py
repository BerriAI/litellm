from datetime import datetime
from typing import Final

from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.llm_cost_calc.typed_token_rates import TokenRates, token_rates
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    PromptTokensDetailsResult,
    billable_completion_tokens,
    billable_prompt_tokens,
    off_peak_window_is_open,
)
from litellm.types.cost_map_configs import ModalityPrices, TokenPricing
from litellm.types.utils import Usage


def _or(price: float | None, fallback: float) -> float:
    return fallback if price is None else price


def _cache_read_cost(tokens: PromptTokensDetailsResult, modality: ModalityPrices, cache_read: float) -> float:
    classified: Final = tokens["cached_text_tokens"] + tokens["cached_audio_tokens"] + tokens["cached_image_tokens"]
    unclassified: Final = max(tokens["cache_hit_tokens"] - classified, 0)
    return (
        (tokens["cached_text_tokens"] + unclassified) * cache_read
        + tokens["cached_audio_tokens"] * _or(modality.cache_read_audio_token, cache_read)
        + tokens["cached_image_tokens"] * _or(modality.cache_read_image_token, cache_read)
    )


def _cache_write_cost(tokens: PromptTokensDetailsResult, rates: TokenRates) -> float:
    details: Final = tokens["cache_creation_token_details"]
    if details is None:
        return tokens["cache_creation_tokens"] * rates.cache_write
    short_ttl: Final = (details.ephemeral_5m_input_tokens or 0) + (details.ephemeral_30m_input_tokens or 0)
    one_hour: Final = details.ephemeral_1h_input_tokens or 0
    twelve_hour: Final = details.ephemeral_12h_input_tokens or 0
    unattributed: Final = max(tokens["cache_creation_tokens"] - short_ttl - one_hour - twelve_hour, 0)
    return (
        (unattributed + short_ttl) * rates.cache_write
        + one_hour * rates.cache_write_1h
        + twelve_hour * rates.cache_write_12h
    )


def typed_cost_per_token(
    pricing: TokenPricing, usage: Usage, current_time: datetime | None = None
) -> tuple[float, float]:
    rates: Final = token_rates(
        pricing,
        usage.prompt_tokens,
        off_peak_is_open=pricing.off_peak is not None
        and off_peak_window_is_open(pricing.off_peak, current_time or current_billing_time()),
    )
    modality: Final = pricing.modality

    prompt: Final = billable_prompt_tokens(usage)
    bills_characters: Final = prompt["character_count"] > 0 and modality.input_character is not None
    bills_images: Final = prompt["image_count"] > 0 and modality.input_image is not None
    prompt_cost: Final = (
        (0 if bills_characters else prompt["text_tokens"]) * rates.input
        + _cache_read_cost(prompt, modality, rates.cache_read)
        + prompt["audio_tokens"] * _or(modality.input_audio_token, rates.input)
        + (0 if bills_images else prompt["image_tokens"]) * _or(modality.input_image_token, rates.input)
        + prompt["video_tokens"] * rates.input
        + _cache_write_cost(prompt, rates)
        + prompt["character_count"] * _or(modality.input_character, 0.0)
        + prompt["image_count"] * _or(modality.input_image, 0.0)
    )

    completion: Final = billable_completion_tokens(usage)
    completion_cost: Final = (
        (completion.text + completion.video) * rates.output
        + completion.reasoning * rates.reasoning
        + completion.audio * _or(modality.output_audio_token, rates.output)
        + completion.image * _or(modality.output_image_token, rates.output)
    )
    return prompt_cost * pricing.multiplier, completion_cost * pricing.multiplier
