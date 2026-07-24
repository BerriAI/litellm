from __future__ import annotations

from dataclasses import dataclass

MICROS_PER_DOLLAR = 1_000_000
TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class PriceQuote:
    input_micros_per_million: int
    cached_input_micros_per_million: int | None
    output_micros_per_million: int | None
    embedding_micros_per_million: int | None


@dataclass(frozen=True, slots=True)
class UsageQuantity:
    input_tokens: int
    cached_input_tokens: int = 0
    output_tokens: int = 0
    embedding: bool = False


def token_charge_micros(tokens: int, rate_micros_per_million: int) -> int:
    if tokens < 0 or rate_micros_per_million < 0:
        raise ValueError("tokens and rates must be non-negative")
    numerator = tokens * rate_micros_per_million
    return (numerator + TOKENS_PER_PRICE_UNIT - 1) // TOKENS_PER_PRICE_UNIT


def calculate_usage_charge(quote: PriceQuote, usage: UsageQuantity) -> int:
    if usage.embedding:
        embedding_rate = quote.embedding_micros_per_million
        if embedding_rate is None:
            raise ValueError("embedding pricing is not configured")
        return token_charge_micros(usage.input_tokens, embedding_rate)

    cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
    uncached_tokens = usage.input_tokens - cached_tokens
    cached_rate = (
        quote.input_micros_per_million
        if quote.cached_input_micros_per_million is None
        else quote.cached_input_micros_per_million
    )
    output_rate = quote.output_micros_per_million
    if usage.output_tokens > 0 and output_rate is None:
        raise ValueError("output pricing is not configured")
    return (
        token_charge_micros(uncached_tokens, quote.input_micros_per_million)
        + token_charge_micros(cached_tokens, cached_rate)
        + token_charge_micros(usage.output_tokens, output_rate or 0)
    )


def display_usd(amount_micros: int) -> str:
    sign = "-" if amount_micros < 0 else ""
    absolute = abs(amount_micros)
    dollars, micros = divmod(absolute, MICROS_PER_DOLLAR)
    return f"{sign}${dollars}.{micros:06d}".rstrip("0").rstrip(".")


def maximum_refund_micros(
    payment_amount_micros: int,
    already_refunded_micros: int,
    available_micros: int,
) -> int:
    if min(payment_amount_micros, already_refunded_micros, available_micros) < 0:
        raise ValueError("refund balances must be non-negative")
    if already_refunded_micros > payment_amount_micros:
        raise ValueError("refunded amount exceeds payment")
    return min(payment_amount_micros - already_refunded_micros, available_micros)
