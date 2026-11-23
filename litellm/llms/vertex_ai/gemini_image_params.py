from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn

from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.llms.vertex_ai.common_utils import map_vertex_image_size

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_FLAT_IMAGE_SETTINGS: Final = MappingProxyType(
    {"aspectRatio": ("aspectRatio", "aspect_ratio"), "imageSize": ("imageSize", "image_size")}
)
IMAGE_CONFIG_ALIASES: Final = ("imageConfig", "image_config")


@dataclass(frozen=True, slots=True)
class UnsupportedImageParam:
    param: str
    reason: str


def raise_unsupported_image_param(rejection: UnsupportedImageParam, model: str) -> NoReturn:
    raise litellm.UnsupportedParamsError(
        model=model,
        llm_provider="vertex_ai",
        message=f"`{rejection.param}` {rejection.reason} for {model}. "
        "To drop it from the call, set `litellm.drop_params = True`.",
    )


def _camel_case(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _camel_case_keys(config: Mapping[str, object]) -> Mapping[str, object]:
    return {
        **{_camel_case(key): value for key, value in config.items() if "_" in key},
        **{key: value for key, value in config.items() if "_" not in key},
    }


def _image_config_object(value: object) -> Mapping[str, object] | None:
    try:
        config: Final = (
            _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
        )
    except ValidationError:
        return None
    return _camel_case_keys(config)


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
    image_config_alias: Final = next((alias for alias in IMAGE_CONFIG_ALIASES if alias in params), None)
    image_config: Final = _EMPTY if image_config_alias is None else _image_config_object(params[image_config_alias])
    size_config: Final = _size_image_config(params["size"], model) if "size" in params else _EMPTY
    if image_config is None and not drop_params:
        return UnsupportedImageParam(image_config_alias or "imageConfig", "must be a JSON object")
    if size_config is None and not drop_params:
        return UnsupportedImageParam("size", "must be 'auto' or '<width>x<height>'")
    explicit_config: Final = image_config or _EMPTY
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
