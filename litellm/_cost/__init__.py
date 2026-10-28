"""Internal pricing API backed by the Rust ``cost`` crate. Not public, its shape may change in any release"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, Literal

from litellm._catalog import Invalid, report_invalid
from litellm.rust_bridge._native import CallContext, ImageOut, Line, Priced, Prices, Unpriced, Usage

Channel = Literal["online", "batch", "decisions"]

_CALL_CUSTOM_PRICING: Final[ContextVar[Mapping[str, object] | None]] = ContextVar(
    "litellm_call_custom_pricing", default=None
)


@contextmanager
def custom_pricing_for_call(custom_pricing: Mapping[str, object] | None) -> Generator[None]:
    """Prices every token meter of the call inside the block with ``custom_pricing`` over its row (decision 12).
    The provider cost calculators keep their signatures, so the overlay reaches them through the call's context"""
    token: Final = _CALL_CUSTOM_PRICING.set(custom_pricing)
    try:
        yield
    finally:
        _CALL_CUSTOM_PRICING.reset(token)


def call_custom_pricing() -> Mapping[str, object] | None:
    return _CALL_CUSTOM_PRICING.get()


def as_main_cost(outcome: Priced | Unpriced | Invalid) -> tuple[float, float]:
    """Main's ``(prompt, completion)`` pair, ``(0.0, 0.0)`` for anything unpriced (decisions 14 and 20)"""
    match outcome:
        case Priced():
            return outcome.prompt, outcome.completion
        case Unpriced():
            return 0.0, 0.0
        case Invalid():
            report_invalid(outcome)
            return 0.0, 0.0


__all__ = [
    "CallContext",
    "Channel",
    "ImageOut",
    "Line",
    "Priced",
    "Prices",
    "Unpriced",
    "Usage",
    "as_main_cost",
    "call_custom_pricing",
    "custom_pricing_for_call",
]
