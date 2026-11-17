from dataclasses import dataclass, replace
from functools import reduce
from typing import Final

from litellm.types.cost_map_configs import LongContextPrices, OffPeakPricing, TierPrices, TokenPrices, TokenPricing


@dataclass(frozen=True, slots=True)
class TokenRates:
    input: float
    output: float
    cache_read: float
    cache_write: float
    cache_write_1h: float
    cache_write_12h: float
    reasoning: float


def _or(price: float | None, fallback: float) -> float:
    return fallback if price is None else price


def _first(preferred: float | None, fallback: float | None) -> float | None:
    return fallback if preferred is None else preferred


def _starting_prices(pricing: TokenPricing) -> TokenPrices:
    standard: Final = pricing.standard
    tier: Final = pricing.service_tier or TokenPrices()
    output: Final = _or(_first(tier.output, standard.output), 0.0)
    return TokenPrices(
        input=_or(_first(tier.input, standard.input), 0.0),
        output=_or(pricing.modality.output_image_token, 0.0) if output == 0.0 else output,
        cache_read=_first(tier.cache_read, standard.cache_read),
        cache_write=_first(tier.cache_write, standard.cache_write),
        cache_write_1h=_first(tier.cache_write_1h, standard.cache_write_1h),
        cache_write_12h=_first(tier.cache_write_12h, standard.cache_write_12h),
    )


def _long_context_prices(pricing: TokenPricing, prompt_tokens: int) -> TokenPrices:
    """Above a long-context threshold a long-TTL cache write only reads the rate of the request's own
    service tier, so a tier with no long-context long-TTL rate keeps its short-context one."""

    def crossed(prices: TokenPrices, band: LongContextPrices) -> TokenPrices:
        tier: Final = TokenPrices() if pricing.service_tier is None else band.service_tier
        long_ttl: Final = band.standard if pricing.service_tier is None else band.service_tier
        return TokenPrices(
            input=_first(tier.input, _first(band.standard.input, prices.input)),
            output=_first(tier.output, _first(band.standard.output, prices.output)),
            cache_read=_first(tier.cache_read, _first(band.standard.cache_read, prices.cache_read)),
            cache_write=_first(tier.cache_write, _first(band.standard.cache_write, prices.cache_write)),
            cache_write_1h=_first(long_ttl.cache_write_1h, prices.cache_write_1h),
            cache_write_12h=_first(long_ttl.cache_write_12h, prices.cache_write_12h),
        )

    return reduce(
        crossed,
        sorted(
            (band for band in pricing.long_context if prompt_tokens > band.above_prompt_tokens),
            key=lambda band: band.above_prompt_tokens,
        ),
        _starting_prices(pricing),
    )


def _priced_tier(tiers: tuple[TierPrices, ...], prompt_tokens: int) -> TokenPrices | None:
    if not tiers or prompt_tokens <= 0:
        return None
    ordered: Final = sorted(tiers, key=lambda tier: tier.above_prompt_tokens)
    selected: Final = next(
        (tier for tier in ordered if tier.above_prompt_tokens < prompt_tokens <= tier.up_to_prompt_tokens),
        ordered[-1],
    )
    return None if selected.prices.input is None else selected.prices


def _tier_prices(tier: TokenPrices, standard: TokenPrices) -> TokenPrices:
    """A tier bills the whole request, so a rate it leaves out is its own input rate. Output is the
    exception, it falls back to the model's output rate so completions are never free by omission."""
    return replace(
        tier,
        output=_or(_first(tier.output, standard.output), 0.0),
        cache_read=_first(tier.cache_read, tier.input),
        cache_write=_first(tier.cache_write, tier.input),
    )


def _reasoning_rate(
    pricing: TokenPricing, tier: TokenPrices | None, off_peak: OffPeakPricing | None, output: float
) -> float:
    if off_peak is not None and off_peak.output_cost_per_reasoning_token is not None:
        return off_peak.output_cost_per_reasoning_token
    tier_rate: Final = None if tier is None else _first(tier.reasoning, tier.output)
    if tier_rate is not None:
        return tier_rate
    service_tier: Final = pricing.service_tier or TokenPrices()
    if service_tier.reasoning is not None:
        return service_tier.reasoning
    return output if service_tier.output is not None else _or(pricing.standard.reasoning, output)


def token_rates(pricing: TokenPricing, prompt_tokens: int, *, off_peak_is_open: bool) -> TokenRates:
    """Off-peak rates replace whatever would otherwise apply, and a cache rate nobody set bills as input."""
    off_peak: Final = pricing.off_peak if off_peak_is_open else None
    off: Final = off_peak or OffPeakPricing()
    tier: Final = _priced_tier(pricing.tiers, prompt_tokens)
    prices: Final = (
        _long_context_prices(pricing, prompt_tokens) if tier is None else _tier_prices(tier, pricing.standard)
    )
    input_rate: Final = _or(prices.input, 0.0)
    unset_cache_rate: Final = _or(off.input_cost_per_token, input_rate)
    output: Final = _or(off.output_cost_per_token, _or(prices.output, 0.0))
    cache_write: Final = _or(off.cache_creation_input_token_cost, _or(prices.cache_write, unset_cache_rate))
    return TokenRates(
        input=_or(off.input_cost_per_token, input_rate),
        output=output,
        cache_read=_or(off.cache_read_input_token_cost, _or(prices.cache_read, unset_cache_rate)),
        cache_write=cache_write,
        cache_write_1h=_or(prices.cache_write_1h, cache_write),
        cache_write_12h=_or(prices.cache_write_12h, cache_write),
        reasoning=_reasoning_rate(pricing, tier, off_peak, output),
    )
