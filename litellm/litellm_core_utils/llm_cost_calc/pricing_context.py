from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm.types.utils import ModelInfo

_MODELS: Final = TypeAdapter(dict[str, ModelInfo])
_DISCOUNTS: Final = TypeAdapter(dict[str, float])
_MARGINS: Final = TypeAdapter(dict[str, float | dict[str, float]])


@dataclass(frozen=True, slots=True)
class RetainedPricing:
    models_json: str
    discounts_json: str
    margins_json: str

    def models(self) -> Mapping[str, ModelInfo]:
        return _MODELS.validate_json(self.models_json)


_CURRENT: Final[ContextVar[RetainedPricing | None]] = ContextVar("retained_pricing", default=None)


@contextmanager
def retained_pricing(pricing: RetainedPricing) -> Generator[None]:
    token: Final = _CURRENT.set(pricing)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def retained_model_info(model: str, provider: str | None) -> ModelInfo | None:
    context: Final = _CURRENT.get()
    if context is None:
        return None
    models: Final = context.models()
    result: Final = models.get(f"{provider}/{model}") or models.get(model)
    if result is None:
        raise ValueError(f"Pricing snapshot has no retained model {model}")
    return result


def retained_models() -> Mapping[str, ModelInfo] | None:
    context: Final = _CURRENT.get()
    return None if context is None else context.models()


def pricing_discounts() -> Mapping[str, float]:
    import litellm

    context: Final = _CURRENT.get()
    return litellm.cost_discount_config if context is None else _DISCOUNTS.validate_json(context.discounts_json)


def pricing_margins() -> Mapping[str, float | Mapping[str, float]]:
    import litellm

    context: Final = _CURRENT.get()
    return litellm.cost_margin_config if context is None else _MARGINS.validate_json(context.margins_json)
