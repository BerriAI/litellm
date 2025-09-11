"""Utility functions for pydantic-settings sources."""

from __future__ import annotations as _annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Optional, cast

from pydantic import BaseModel, Json, RootModel, Secret
from pydantic._internal._utils import is_model_class
from pydantic.dataclasses import is_pydantic_dataclass
from typing_extensions import get_args, get_origin
from typing_inspection import typing_objects

from ..exceptions import SettingsError
from ..utils import _lenient_issubclass
from .types import EnvNoneType


def _get_env_var_key(key: str, case_sensitive: bool = False) -> str:
    return key if case_sensitive else key.lower()


def _parse_env_none_str(
    value: str | None, parse_none_str: str | None = None
) -> str | None | EnvNoneType:
    return (
        value
        if not (value == parse_none_str and parse_none_str is not None)
        else EnvNoneType(value)
    )


def parse_env_vars(
    env_vars: Mapping[str, str | None],
    case_sensitive: bool = False,
    ignore_empty: bool = False,
    parse_none_str: str | None = None,
) -> Mapping[str, str | None]:
    return {
        _get_env_var_key(k, case_sensitive): _parse_env_none_str(v, parse_none_str)
        for k, v in env_vars.items()
        if not (ignore_empty and v == "")
    }


def _annotation_is_complex(annotation: type[Any] | None, metadata: list[Any]) -> bool:
    # If the model is a root model, the root annotation should be used to
    # evaluate the complexity.
    if (
        annotation is not None
        and _lenient_issubclass(annotation, RootModel)
        and annotation is not RootModel
    ):
        annotation = cast("type[RootModel[Any]]", annotation)
        root_annotation = annotation.model_fields["root"].annotation
        if root_annotation is not None:  # pragma: no branch
            annotation = root_annotation

    if any(isinstance(md, Json) for md in metadata):  # type: ignore[misc]
        return False

    origin = get_origin(annotation)

    # Check if annotation is of the form Annotated[type, metadata].
    if typing_objects.is_annotated(origin):
        # Return result of recursive call on inner type.
        inner, *meta = get_args(annotation)
        return _annotation_is_complex(inner, meta)

    if origin is Secret:
        return False

    return (
        _annotation_is_complex_inner(annotation)
        or _annotation_is_complex_inner(origin)
        or hasattr(origin, "__pydantic_core_schema__")
        or hasattr(origin, "__get_pydantic_core_schema__")
    )


def _annotation_is_complex_inner(annotation: type[Any] | None) -> bool:
    if _lenient_issubclass(annotation, (str, bytes)):
        return False

    return _lenient_issubclass(
        annotation, (BaseModel, Mapping, Sequence, tuple, set, frozenset, deque)
    ) or is_dataclass(annotation)


def _union_is_complex(annotation: type[Any] | None, metadata: list[Any]) -> bool:
    """Check if a union type contains any complex types."""
    return any(_annotation_is_complex(arg, metadata) for arg in get_args(annotation))


def _annotation_contains_types(
    annotation: type[Any] | None,
    types: tuple[Any, ...],
    is_include_origin: bool = True,
    is_strip_annotated: bool = False,
) -> bool:
    """Check if a type annotation contains any of the specified types."""
    if is_strip_annotated:
        annotation = _strip_annotated(annotation)
    if is_include_origin is True and get_origin(annotation) in types:
        return True
    for type_ in get_args(annotation):
        if _annotation_contains_types(
            type_, types, is_include_origin=True, is_strip_annotated=is_strip_annotated
        ):
            return True
    return annotation in types


def _strip_annotated(annotation: Any) -> Any:
    if typing_objects.is_annotated(get_origin(annotation)):
        return annotation.__origin__
    else:
        return annotation


def _annotation_enum_val_to_name(
    annotation: type[Any] | None, value: Any
) -> Optional[str]:
    for type_ in (annotation, get_origin(annotation), *get_args(annotation)):
        if _lenient_issubclass(type_, Enum):
            if value in tuple(val.value for val in type_):
                return type_(value).name
    return None


def _annotation_enum_name_to_val(annotation: type[Any] | None, name: Any) -> Any:
    for type_ in (annotation, get_origin(annotation), *get_args(annotation)):
        if _lenient_issubclass(type_, Enum):
            if name in tuple(val.name for val in type_):
                return type_[name]
    return None


def _get_model_fields(model_cls: type[Any]) -> dict[str, Any]:
    """Get fields from a pydantic model or dataclass."""

    if is_pydantic_dataclass(model_cls) and hasattr(model_cls, "__pydantic_fields__"):
        return model_cls.__pydantic_fields__
    if is_model_class(model_cls):
        return model_cls.model_fields
    raise SettingsError(
        f"Error: {model_cls.__name__} is not subclass of BaseModel or pydantic.dataclasses.dataclass"
    )


def _get_alias_names(
    field_name: str,
    field_info: Any,
    alias_path_args: dict[str, str] = {},
    case_sensitive: bool = True,
) -> tuple[tuple[str, ...], bool]:
    """Get alias names for a field, handling alias paths and case sensitivity."""
    from pydantic import AliasChoices, AliasPath

    alias_names: list[str] = []
    is_alias_path_only: bool = True
    if not any((field_info.alias, field_info.validation_alias)):
        alias_names += [field_name]
        is_alias_path_only = False
    else:
        new_alias_paths: list[AliasPath] = []
        for alias in (field_info.alias, field_info.validation_alias):
            if alias is None:
                continue
            elif isinstance(alias, str):
                alias_names.append(alias)
                is_alias_path_only = False
            elif isinstance(alias, AliasChoices):
                for name in alias.choices:
                    if isinstance(name, str):
                        alias_names.append(name)
                        is_alias_path_only = False
                    else:
                        new_alias_paths.append(name)
            else:
                new_alias_paths.append(alias)
        for alias_path in new_alias_paths:
            name = cast(str, alias_path.path[0])
            name = name.lower() if not case_sensitive else name
            alias_path_args[name] = "dict" if len(alias_path.path) > 2 else "list"
            if not alias_names and is_alias_path_only:
                alias_names.append(name)
    if not case_sensitive:
        alias_names = [alias_name.lower() for alias_name in alias_names]
    return tuple(dict.fromkeys(alias_names)), is_alias_path_only


def _is_function(obj: Any) -> bool:
    """Check if an object is a function."""
    from types import BuiltinFunctionType, FunctionType

    return isinstance(obj, (FunctionType, BuiltinFunctionType))


__all__ = [
    "_annotation_contains_types",
    "_annotation_enum_name_to_val",
    "_annotation_enum_val_to_name",
    "_annotation_is_complex",
    "_annotation_is_complex_inner",
    "_get_alias_names",
    "_get_env_var_key",
    "_get_model_fields",
    "_is_function",
    "_parse_env_none_str",
    "_strip_annotated",
    "_union_is_complex",
    "parse_env_vars",
]
