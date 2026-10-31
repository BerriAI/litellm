import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

_RATES: Final = TypeAdapter(dict[str, object])


class Currency(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(pattern=r"^[A-Z]{3}$")
    usd_rate: float = Field(gt=0, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class CurrencyConfigError:
    message: str


class RateSource(ABC):
    @abstractmethod
    def get_currency(self, code: str) -> Currency | CurrencyConfigError: ...


@dataclass(frozen=True, slots=True)
class FixedRates(RateSource):
    rates: tuple[Currency, ...]

    def get_currency(self, code: str) -> Currency | CurrencyConfigError:
        if code == "USD":
            return Currency(code="USD", usd_rate=1)
        return next(
            (rate for rate in self.rates if rate.code == code), CurrencyConfigError(f"Missing exchange rate for {code}")
        )


def _valid_code(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value) is not None


def _parse_rate(code: str, value: object) -> Currency | CurrencyConfigError:
    if not _valid_code(code) or isinstance(value, bool) or not isinstance(value, (int, float)):
        return CurrencyConfigError("exchange_rates must contain uppercase currency codes and positive finite rates")
    try:
        if value <= 0 or not math.isfinite(value):
            return CurrencyConfigError("exchange_rates must contain uppercase currency codes and positive finite rates")
    except OverflowError:
        return CurrencyConfigError("exchange_rates must contain uppercase currency codes and positive finite rates")
    if code == "USD" and value != 1:
        return CurrencyConfigError("The USD exchange rate must be 1")
    return Currency(code=code, usd_rate=value)


def currency_from_settings(settings: Mapping[str, object]) -> Currency | CurrencyConfigError:
    code: Final = settings.get("display_currency", "USD")
    rates: Final = settings.get("exchange_rates", {})
    if not _valid_code(code) or not isinstance(code, str):
        return CurrencyConfigError("display_currency must be a three-letter uppercase currency code")
    if not isinstance(rates, Mapping):
        return CurrencyConfigError("exchange_rates must be a mapping of currency codes to positive finite rates")
    try:
        parsed: Final = _RATES.validate_python(rates)
    except ValidationError:
        return CurrencyConfigError("exchange_rates must have currency codes as keys")
    entries: Final = tuple(_parse_rate(rate_code, rate) for rate_code, rate in parsed.items())
    for entry in entries:
        if isinstance(entry, CurrencyConfigError):
            return entry
    source: Final = FixedRates(tuple(entry for entry in entries if isinstance(entry, Currency)))
    return source.get_currency(code)


def _raise_currency_error(error: CurrencyConfigError) -> NoReturn:
    raise ValueError(f"Invalid currency configuration: {error.message}")


def require_currency(settings: Mapping[str, object]) -> Currency:
    result: Final = currency_from_settings(settings)
    if isinstance(result, CurrencyConfigError):
        _raise_currency_error(result)
    return result
