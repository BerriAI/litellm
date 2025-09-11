"""Type definitions for pydantic-settings sources."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union

if TYPE_CHECKING:
    from pydantic._internal._dataclasses import PydanticDataclass
    from pydantic.main import BaseModel

    PydanticModel = TypeVar("PydanticModel", bound=Union[PydanticDataclass, BaseModel])
else:
    PydanticModel = Any


class EnvNoneType(str):
    pass


class NoDecode:
    """Annotation to prevent decoding of a field value."""

    pass


class ForceDecode:
    """Annotation to force decoding of a field value."""

    pass


DotenvType = Union[Path, str, Sequence[Union[Path, str]]]
PathType = Union[Path, str, Sequence[Union[Path, str]]]
DEFAULT_PATH: PathType = Path("")

# This is used as default value for `_env_file` in the `BaseSettings` class and
# `env_file` in `DotEnvSettingsSource` so the default can be distinguished from `None`.
# See the docstring of `BaseSettings` for more details.
ENV_FILE_SENTINEL: DotenvType = Path("")


class _CliSubCommand:
    pass


class _CliPositionalArg:
    pass


class _CliImplicitFlag:
    pass


class _CliExplicitFlag:
    pass


class _CliUnknownArgs:
    pass


__all__ = [
    "DEFAULT_PATH",
    "ENV_FILE_SENTINEL",
    "DotenvType",
    "EnvNoneType",
    "ForceDecode",
    "NoDecode",
    "PathType",
    "PydanticModel",
    "_CliExplicitFlag",
    "_CliImplicitFlag",
    "_CliPositionalArg",
    "_CliSubCommand",
    "_CliUnknownArgs",
]
