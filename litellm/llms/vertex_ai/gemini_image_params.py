from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final, NoReturn

from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import camel_case_keys
from litellm.llms.vertex_ai.common_utils import map_vertex_image_size
from litellm.types.images.main import IMAGE_CONFIG_FIELDS, UnsupportedImageParam

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_FLAT_IMAGE_SETTINGS: Final = MappingProxyType(
    {"aspectRatio": ("aspectRatio", "aspect_ratio"), "imageSize": ("imageSize", "image_size")}
)


def raise_unsupported_image_param(rejection: UnsupportedImageParam, model: str) -> NoReturn:
    raise litellm.UnsupportedParamsError(
        model=model,
        llm_provider="vertex_ai",
        message=f"`{rejection.param}` {rejection.reason} for {model}. "
        "To drop it from the call, set `litellm.drop_params = True`.",
    )


def _image_config_object(value: object) -> Mapping[str, object] | None:
    try:
        config: Final = (
            _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
        )
    except ValidationError:
        return None
    return camel_case_keys(config)


def _size_image_config(size: object, model: str) -> Mapping[str, object] | None:
    return _EMPTY if size == "auto" else map_vertex_image_size(size, model)


def _flat_image_settings(params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        field: next(params[alias] for alias in aliases if alias in params)
        for field, aliases in _FLAT_IMAGE_SETTINGS.items()
        if any(alias in params for alias in aliases)
    }


def map_vertex_gemini_image_params(
    params: Mapping[str, object], model: str, drop_params: bool
) -> Mapping[str, object] | UnsupportedImageParam:
    image_configs: Final = tuple(
        (alias, _image_config_object(params[alias])) for alias in IMAGE_CONFIG_FIELDS if alias in params
    )
    unmappable_alias: Final = next((alias for alias, config in image_configs if config is None), None)
    size_config: Final = _size_image_config(params["size"], model) if "size" in params else _EMPTY
    if unmappable_alias is not None and not drop_params:
        return UnsupportedImageParam(unmappable_alias, "must be a JSON object")
    if size_config is None and not drop_params:
        return UnsupportedImageParam("size", "must be 'auto' or '<width>x<height>'")
    mapped_configs: Final = tuple(config for _, config in image_configs if config is not None)
    explicit_config: Final[Mapping[str, object]] = dict(
        chain.from_iterable(config.items() for config in reversed(mapped_configs))
    )
    return {
        **{key: value for key, value in (size_config or _EMPTY).items() if key not in explicit_config},
        **_flat_image_settings(params),
        **({"imageConfig": dict(explicit_config)} if explicit_config else {}),
    }


def vertex_gemini_image_config(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        **(_image_config_object(optional_params.get("imageConfig") or {}) or _EMPTY),
        **_flat_image_settings(optional_params),
    }
