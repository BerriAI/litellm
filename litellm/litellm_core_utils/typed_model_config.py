from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Final, Literal, overload

from pydantic import ValidationError

from litellm._logging import verbose_logger
from litellm.types.cost_map_configs import (
    AnthropicModelOnAnthropicProvider,
    OpenAIModelOnOpenAIProvider,
    SharedModelConfig,
)
from litellm.types.utils import ModelInfo

_CONFIGS: Final[Mapping[str, type[SharedModelConfig]]] = MappingProxyType(
    {
        OpenAIModelOnOpenAIProvider.inference_provider: OpenAIModelOnOpenAIProvider,
        AnthropicModelOnAnthropicProvider.inference_provider: AnthropicModelOnAnthropicProvider,
    }
)
_COST_MAP_FIELDS: Final = frozenset(ModelInfo.__required_keys__ | ModelInfo.__optional_keys__).union(
    *(config.model_fields for config in _CONFIGS.values())
)


def _is_cost_map_field(key: str) -> bool:
    return key in _COST_MAP_FIELDS or key.startswith("supports_")


@lru_cache(maxsize=1024)
def _warn_once(message: str, config: str, model: str, fields: tuple[str, ...]) -> None:
    verbose_logger.warning(message, config, model, ", ".join(fields))


def _catalog_row(model: str, custom_llm_provider: str) -> Mapping[str, object]:
    import litellm

    try:
        key: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)["key"]
    except Exception:  # noqa: BLE001  # an unmapped model has no catalog row
        return MappingProxyType({})
    return litellm.model_cost.get(key, {})


def _parse(config: type[SharedModelConfig], model: str, row: Mapping[str, object]) -> SharedModelConfig:
    try:
        return config.model_validate(row)
    except ValidationError as error:
        invalid: Final = tuple(sorted({str(detail["loc"][0]) for detail in error.errors()}))
        _warn_once("%s: model=%s has invalid values for %s, these fields are ignored", config.__name__, model, invalid)
        return config.model_validate({key: value for key, value in row.items() if key not in invalid})


@overload
def typed_model_config(
    model: str, custom_llm_provider: Literal["openai"], model_info: Mapping[str, object] | None = None
) -> OpenAIModelOnOpenAIProvider: ...


@overload
def typed_model_config(
    model: str, custom_llm_provider: Literal["anthropic"], model_info: Mapping[str, object] | None = None
) -> AnthropicModelOnAnthropicProvider: ...


def typed_model_config(
    model: str, custom_llm_provider: Literal["openai", "anthropic"], model_info: Mapping[str, object] | None = None
) -> SharedModelConfig:
    config: Final = _CONFIGS[custom_llm_provider]
    row: Final = {**_catalog_row(model, custom_llm_provider), **(model_info or {})}
    unused: Final = tuple(sorted(key for key in row if key not in config.model_fields and _is_cost_map_field(key)))
    if unused:
        _warn_once("%s: model=%s sets %s, which this provider does not use", config.__name__, model, unused)
    return _parse(config, model, row)
