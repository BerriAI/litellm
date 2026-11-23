from collections.abc import Collection, Iterator, Mapping
from functools import partial, reduce
from io import BufferedReader, BytesIO
from itertools import chain
from typing import Any, Final, cast, get_type_hints

from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import camel_case_keys, drops_unsupported_params
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.files import FILE_MIME_TYPES, FileType
from litellm.types.images.main import ImageEditOptionalRequestParams, UnsupportedImageParam

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_MAX_BRACKET_DEPTH: Final = 4
_MAX_BRACKET_FIELDS: Final = 64


def _bracket_path(key: str, prefix: str) -> tuple[str, ...]:
    path: Final = tuple(key[len(prefix) : -1].split("][")) if key.endswith("]") else ()
    well_formed: Final = all(segment and "[" not in segment and "]" not in segment for segment in path)
    return path if well_formed and len(path) <= _MAX_BRACKET_DEPTH else ()


def _strict_prefixes(path: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    return (path[:depth] for depth in range(1, len(path)))


def _nested_value(entries: tuple[tuple[tuple[str, ...], object], ...]) -> object:
    deeper: Final = tuple(entry for entry in entries if entry[0])
    return _nest_bracket_entries(deeper) if deeper else entries[-1][1]


def _camel_case_tree(config: Mapping[str, object], depth: int = 0) -> Mapping[str, object]:
    return camel_case_keys({key: _camel_case_subtree(value, depth + 1) for key, value in config.items()})


def _camel_case_subtree(value: object, depth: int) -> object:
    if depth >= _MAX_BRACKET_DEPTH or not isinstance(value, Mapping):
        return value
    return _camel_case_tree(_JSON_OBJECT.validate_python(value), depth)


def _nest_bracket_entries(entries: tuple[tuple[tuple[str, ...], object], ...]) -> Mapping[str, object]:
    return {
        head: _nested_value(tuple((path[1:], value) for path, value in entries if path[0] == head))
        for head in dict.fromkeys(path[0] for path, _ in entries)
    }


def _merged_value(base: object, override: object) -> object:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        return _deep_merge(_JSON_OBJECT.validate_python(base), _JSON_OBJECT.validate_python(override))
    return override


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> Mapping[str, object]:
    return {**base, **{key: _merged_value(base.get(key), value) for key, value in override.items()}}


def _json_object_or_none(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _fold_bracket_form_field(
    params: Mapping[str, object] | UnsupportedImageParam, field: str, drop_params: bool
) -> Mapping[str, object] | UnsupportedImageParam:
    if isinstance(params, UnsupportedImageParam):
        return params
    prefix: Final = f"{field}["
    bracket_keys: Final = tuple(key for key in params if key.startswith(prefix))
    if not bracket_keys:
        return params
    rest: Final = {key: value for key, value in params.items() if not key.startswith(prefix)}
    if len(bracket_keys) > _MAX_BRACKET_FIELDS:
        return rest if drop_params else UnsupportedImageParam(field, f"takes at most {_MAX_BRACKET_FIELDS} form fields")
    paths: Final = tuple((_bracket_path(key, prefix), key) for key in bracket_keys)
    malformed: Final = next((key for path, key in paths if not path), None)
    if malformed is not None and not drop_params:
        return UnsupportedImageParam(malformed, f"must be `{field}[key]` nested at most {_MAX_BRACKET_DEPTH} levels")
    entries: Final = tuple((path, params[key]) for path, key in paths if path)
    nested_paths: Final = frozenset(chain.from_iterable(_strict_prefixes(path) for path, _ in entries))
    clash: Final = next((key for path, key in paths if path in nested_paths), None)
    if clash is not None and not drop_params:
        return UnsupportedImageParam(clash, "sets a value where other form fields nest an object")
    sent: Final = params.get(field)
    base: Final[Mapping[str, object] | None] = {} if sent is None else _json_object_or_none(sent)
    if base is None and not drop_params:
        return UnsupportedImageParam(field, "must be a JSON object")
    if not entries:
        return rest
    return {**rest, field: _deep_merge(_camel_case_tree(base or {}), _camel_case_tree(_nest_bracket_entries(entries)))}


class ImageEditRequestUtils:
    @staticmethod
    def get_optional_params_image_edit(
        model: str,
        image_edit_provider_config: BaseImageEditConfig,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        drop_params: object = None,
        additional_drop_params: Collection[str] | None = None,
    ) -> dict:
        """
        Get optional parameters for the image edit API.

        Args:
            model: The model name
            image_edit_provider_config: The provider configuration for image edit API
            image_edit_optional_params: The optional parameters for the image edit API
            drop_params: If True, silently drop unsupported parameters instead of raising
            additional_drop_params: List of additional parameter names to drop

        Returns:
            A dictionary of supported parameters for the image edit API
        """
        supported_params: Final = image_edit_provider_config.get_supported_openai_params(model)

        should_drop: Final = drops_unsupported_params(drop_params)

        filtered_optional_params: Final = dict(image_edit_optional_params)
        if additional_drop_params:
            for param in additional_drop_params:
                filtered_optional_params.pop(param, None)

        unsupported_params: Final = [param for param in filtered_optional_params if param not in supported_params]

        if unsupported_params:
            if should_drop:
                for param in unsupported_params:
                    filtered_optional_params.pop(param, None)
            else:
                raise litellm.UnsupportedParamsError(
                    model=model,
                    message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
                )

        mapped_params: Final = image_edit_provider_config.map_openai_params(
            image_edit_optional_params=cast(ImageEditOptionalRequestParams, filtered_optional_params),
            model=model,
            drop_params=should_drop,
        )

        return mapped_params

    @staticmethod
    def fold_bracket_form_fields(
        params: Mapping[str, object], fields: Collection[str], drop_params: bool
    ) -> Mapping[str, object] | UnsupportedImageParam:
        initial: Final[Mapping[str, object] | UnsupportedImageParam] = params
        return reduce(partial(_fold_bracket_form_field, drop_params=drop_params), fields, initial)

    @staticmethod
    def get_requested_image_edit_optional_param(
        params: Mapping[str, object],
        provider_supported_params: Collection[str] = (),
    ) -> ImageEditOptionalRequestParams:
        """
        Filter parameters to only include those defined in ImageEditOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ImageEditOptionalRequestParams instance with only the valid parameters
        """
        valid_keys: Final = frozenset(get_type_hints(ImageEditOptionalRequestParams)) | frozenset(
            provider_supported_params
        )
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        return cast(ImageEditOptionalRequestParams, filtered_params)

    @staticmethod
    def get_image_content_type(image_data: Any) -> str:
        """
        Detect the content type of image data using existing LiteLLM utils.

        Args:
            image_data: Can be BytesIO, bytes, BufferedReader, or other file-like objects

        Returns:
            The MIME type string (e.g., "image/png", "image/jpeg")
        """
        try:
            # Extract bytes for content type detection
            if isinstance(image_data, BytesIO):
                # Save current position
                current_pos = image_data.tell()
                image_data.seek(0)
                bytes_data = image_data.read(100)  # First 100 bytes are enough for detection
                # Restore position
                image_data.seek(current_pos)
            elif isinstance(image_data, BufferedReader):
                # Save current position
                current_pos = image_data.tell()
                image_data.seek(0)
                bytes_data = image_data.read(100)
                # Restore position
                image_data.seek(current_pos)
            elif isinstance(image_data, bytes):
                bytes_data = image_data[:100]
            else:
                # For other types, try to read if possible
                if hasattr(image_data, "read"):
                    current_pos = getattr(image_data, "tell", lambda: 0)()
                    if hasattr(image_data, "seek"):
                        image_data.seek(0)
                    bytes_data = image_data.read(100)
                    if hasattr(image_data, "seek"):
                        image_data.seek(current_pos)
                else:
                    return FILE_MIME_TYPES[FileType.PNG]  # Default fallback

            # Use the existing get_image_type function to detect image type
            image_type_str: Final = get_image_type(bytes_data)

            if image_type_str is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default if detection fails

            # Map detected type string to FileType enum and get MIME type
            type_mapping: Final = {
                "png": FileType.PNG,
                "jpeg": FileType.JPEG,
                "gif": FileType.GIF,
                "webp": FileType.WEBP,
                "heic": FileType.HEIC,
            }

            file_type: Final = type_mapping.get(image_type_str)
            if file_type is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default to PNG if unknown

            return FILE_MIME_TYPES[file_type]

        except Exception:
            # If anything goes wrong, default to PNG
            return FILE_MIME_TYPES[FileType.PNG]
