from collections import ChainMap
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # live catalog has no typed boundary yet

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
    allow_live_models: bool = False

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
    if result is None and not context.allow_live_models:
        raise ValueError(f"Pricing snapshot has no retained model {model}")
    return result


def retained_models() -> Mapping[str, ModelInfo] | None:
    import litellm

    context: Final = _CURRENT.get()
    if context is None:
        return None
    if not context.allow_live_models:
        return context.models()
    catalog: Final = cast(  # cast-ok: legacy catalog getter has an untyped dict signature
        dict[str, ModelInfo], vars(litellm)["model_cost"]
    )
    return ChainMap(dict(context.models()), catalog)


def pricing_discounts() -> Mapping[str, float]:
    import litellm

    context: Final = _CURRENT.get()
    return (
        litellm.cost_discount_config
        if context is None or context.allow_live_models
        else _DISCOUNTS.validate_json(context.discounts_json)
    )


def pricing_margins() -> Mapping[str, float | Mapping[str, float]]:
    import litellm

    context: Final = _CURRENT.get()
    return (
        litellm.cost_margin_config
        if context is None or context.allow_live_models
        else _MARGINS.validate_json(context.margins_json)
    )
