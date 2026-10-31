from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.spend_tracking.currency import (
    Currency,
    CurrencyConfigError,
    FixedRates,
    currency_from_settings,
    require_currency,
)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, Currency(code="USD", usd_rate=1)),
        ({"display_currency": "EUR", "exchange_rates": {"EUR": 0.92}}, Currency(code="EUR", usd_rate=0.92)),
        ({"display_currency": "USD", "exchange_rates": {"EUR": 0.92}}, Currency(code="USD", usd_rate=1)),
    ],
)
def test_currency_from_settings(settings: dict[str, object], expected: Currency) -> None:
    assert currency_from_settings(settings) == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"display_currency": "EUR"},
        {"display_currency": "eur", "exchange_rates": {"eur": 1}},
        {"display_currency": "EUR\n", "exchange_rates": {"EUR": 1}},
        {"display_currency": None},
        {"exchange_rates": None},
        {"exchange_rates": {"EUR": 0}},
        {"exchange_rates": {"EUR": -1}},
        {"exchange_rates": {"EUR": float("nan")}},
        {"exchange_rates": {"EUR": float("inf")}},
        {"exchange_rates": {"EUR": True}},
        {"exchange_rates": {"EUR": "0.92"}},
        {"exchange_rates": {"USD": 2}},
        {"exchange_rates": {"INVALID": 1}},
    ],
)
def test_invalid_currency_config_is_an_error(settings: dict[str, object]) -> None:
    assert isinstance(currency_from_settings(settings), CurrencyConfigError)
    with pytest.raises(ValueError, match="Invalid currency configuration"):
        require_currency(settings)


def test_fixed_rates_returns_selected_rate_and_reports_missing_rate() -> None:
    source: Final = FixedRates((Currency(code="EUR", usd_rate=0.92),))
    assert source.get_currency("EUR") == Currency(code="EUR", usd_rate=0.92)
    assert source.get_currency("USD") == Currency(code="USD", usd_rate=1)
    assert isinstance(source.get_currency("GBP"), CurrencyConfigError)


def test_currency_is_frozen_and_validated() -> None:
    currency: Final = Currency(code="EUR", usd_rate=0.92)
    with pytest.raises(ValidationError):
        setattr(currency, "usd_rate", 2)
    with pytest.raises(ValidationError):
        Currency(code="EUR", usd_rate=0)
