"""Dotenv file settings source."""

from __future__ import annotations as _annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import dotenv_values
from pydantic._internal._typing_extra import (  # type: ignore[attr-defined]
    get_origin,
)
from typing_inspection.introspection import is_union_origin

from ..types import ENV_FILE_SENTINEL, DotenvType
from ..utils import (
    _annotation_is_complex,
    _union_is_complex,
    parse_env_vars,
)
from .env import EnvSettingsSource

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings


class DotEnvSettingsSource(EnvSettingsSource):
    """
    Source class for loading settings values from env files.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        env_file: DotenvType | None = ENV_FILE_SENTINEL,
        env_file_encoding: str | None = None,
        case_sensitive: bool | None = None,
        env_prefix: str | None = None,
        env_nested_delimiter: str | None = None,
        env_nested_max_split: int | None = None,
        env_ignore_empty: bool | None = None,
        env_parse_none_str: str | None = None,
        env_parse_enums: bool | None = None,
    ) -> None:
        self.env_file = (
            env_file
            if env_file != ENV_FILE_SENTINEL
            else settings_cls.model_config.get("env_file")
        )
        self.env_file_encoding = (
            env_file_encoding
            if env_file_encoding is not None
            else settings_cls.model_config.get("env_file_encoding")
        )
        super().__init__(
            settings_cls,
            case_sensitive,
            env_prefix,
            env_nested_delimiter,
            env_nested_max_split,
            env_ignore_empty,
            env_parse_none_str,
            env_parse_enums,
        )

    def _load_env_vars(self) -> Mapping[str, str | None]:
        return self._read_env_files()

    @staticmethod
    def _static_read_env_file(
        file_path: Path,
        *,
        encoding: str | None = None,
        case_sensitive: bool = False,
        ignore_empty: bool = False,
        parse_none_str: str | None = None,
    ) -> Mapping[str, str | None]:
        file_vars: dict[str, str | None] = dotenv_values(
            file_path, encoding=encoding or "utf8"
        )
        return parse_env_vars(file_vars, case_sensitive, ignore_empty, parse_none_str)

    def _read_env_file(
        self,
        file_path: Path,
    ) -> Mapping[str, str | None]:
        return self._static_read_env_file(
            file_path,
            encoding=self.env_file_encoding,
            case_sensitive=self.case_sensitive,
            ignore_empty=self.env_ignore_empty,
            parse_none_str=self.env_parse_none_str,
        )

    def _read_env_files(self) -> Mapping[str, str | None]:
        env_files = self.env_file
        if env_files is None:
            return {}

        if isinstance(env_files, (str, os.PathLike)):
            env_files = [env_files]

        dotenv_vars: dict[str, str | None] = {}
        for env_file in env_files:
            env_path = Path(env_file).expanduser()
            if env_path.is_file():
                dotenv_vars.update(self._read_env_file(env_path))

        return dotenv_vars

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = super().__call__()
        is_extra_allowed = self.config.get("extra") != "forbid"

        # As `extra` config is allowed in dotenv settings source, We have to
        # update data with extra env variables from dotenv file.
        for env_name, env_value in self.env_vars.items():
            if not env_value or env_name in data:
                continue
            env_used = False
            for field_name, field in self.settings_cls.model_fields.items():
                for _, field_env_name, _ in self._extract_field_info(field, field_name):
                    if env_name == field_env_name or (
                        (
                            _annotation_is_complex(field.annotation, field.metadata)
                            or (
                                is_union_origin(get_origin(field.annotation))
                                and _union_is_complex(field.annotation, field.metadata)
                            )
                        )
                        and env_name.startswith(field_env_name)
                    ):
                        env_used = True
                        break
                if env_used:
                    break
            if not env_used:
                if is_extra_allowed and env_name.startswith(self.env_prefix):
                    # env_prefix should be respected and removed from the env_name
                    normalized_env_name = env_name[len(self.env_prefix) :]
                    data[normalized_env_name] = env_value
                else:
                    data[env_name] = env_value
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env_file={self.env_file!r}, env_file_encoding={self.env_file_encoding!r}, "
            f"env_nested_delimiter={self.env_nested_delimiter!r}, env_prefix_len={self.env_prefix_len!r})"
        )


def read_env_file(
    file_path: Path,
    *,
    encoding: str | None = None,
    case_sensitive: bool = False,
    ignore_empty: bool = False,
    parse_none_str: str | None = None,
) -> Mapping[str, str | None]:
    warnings.warn(
        "read_env_file will be removed in the next version, use DotEnvSettingsSource._static_read_env_file if you must",
        DeprecationWarning,
    )
    return DotEnvSettingsSource._static_read_env_file(
        file_path,
        encoding=encoding,
        case_sensitive=case_sensitive,
        ignore_empty=ignore_empty,
        parse_none_str=parse_none_str,
    )


__all__ = ["DotEnvSettingsSource", "read_env_file"]
