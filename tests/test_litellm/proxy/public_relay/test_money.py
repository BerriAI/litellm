import pytest

from litellm.proxy.public_relay.money import (
    PriceQuote,
    UsageQuantity,
    calculate_usage_charge,
    display_usd,
    token_charge_micros,
)


@pytest.mark.parametrize(
    ("tokens", "rate", "expected"),
    [
        (0, 1_000_000, 0),
        (1, 1, 1),
        (1, 2_000_000, 2),
        (1_000_001, 1, 2),
    ],
)
def test_token_charge_rounds_up(tokens: int, rate: int, expected: int) -> None:
    assert token_charge_micros(tokens, rate) == expected


def test_chat_charge_splits_cached_input_and_output() -> None:
    quote = PriceQuote(
        input_micros_per_million=2_000_000,
        cached_input_micros_per_million=500_000,
        output_micros_per_million=8_000_000,
        embedding_micros_per_million=None,
    )

    charge = calculate_usage_charge(
        quote,
        UsageQuantity(input_tokens=1_000, cached_input_tokens=250, output_tokens=100),
    )

    assert charge == 2_425


def test_zero_cached_input_rate_remains_free() -> None:
    quote = PriceQuote(
        input_micros_per_million=2_000_000,
        cached_input_micros_per_million=0,
        output_micros_per_million=8_000_000,
        embedding_micros_per_million=None,
    )

    assert calculate_usage_charge(quote, UsageQuantity(input_tokens=1_000, cached_input_tokens=1_000)) == 0


def test_embedding_uses_embedding_rate_only() -> None:
    quote = PriceQuote(
        input_micros_per_million=9_000_000,
        cached_input_micros_per_million=None,
        output_micros_per_million=None,
        embedding_micros_per_million=100_000,
    )

    assert calculate_usage_charge(quote, UsageQuantity(input_tokens=12_345, embedding=True)) == 1_235


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0, "$0"),
        (1, "$0.000001"),
        (1_250_000, "$1.25"),
        (-20_000_000, "-$20"),
    ],
)
def test_display_usd(amount: int, expected: str) -> None:
    assert display_usd(amount) == expected
