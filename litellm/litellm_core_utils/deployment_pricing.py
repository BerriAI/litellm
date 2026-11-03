from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_STR_KEYED: Final = TypeAdapter(dict[str, object])


def pricing_models(base_model: object, model: object) -> tuple[str, ...]:
    """Cost-map keys to price a deployment under, ``model_info.base_model`` first, then the
    deployment's own model, blanks and duplicates dropped. Every spend and routing surface that
    prices a deployment walks this order, so a ``base_model`` the cost map does not know falls
    through to the deployment model instead of billing or ranking at the unmapped default."""
    return tuple(dict.fromkeys(_named_models(base_model, model)))


def deployment_pricing_models(deployment: object) -> tuple[str, ...]:
    """``pricing_models`` read off a raw deployment dict (``model_info.base_model``, then
    ``litellm_params.base_model``, then ``litellm_params.model``); malformed blocks count as unset."""
    view: Final = str_keyed(deployment)
    params: Final = str_keyed(view.get("litellm_params"))
    info: Final = str_keyed(view.get("model_info"))
    return pricing_models(info.get("base_model") or params.get("base_model"), params.get("model"))


def _named_models(*candidates: object) -> Iterator[str]:
    return (candidate for candidate in candidates if isinstance(candidate, str) and candidate)


def str_keyed(value: object) -> Mapping[str, object]:
    """``value`` as a read-only str-keyed mapping, empty when it is not a dict."""
    try:
        return MappingProxyType(_STR_KEYED.validate_python(value))
    except ValidationError:
        return _EMPTY
