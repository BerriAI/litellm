"""Secrets file settings source."""

from __future__ import annotations as _annotations

import os
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic.fields import FieldInfo

from pydantic_settings.utils import path_type_label

from ...exceptions import SettingsError
from ..base import PydanticBaseEnvSettingsSource
from ..types import PathType

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings


class SecretsSettingsSource(PydanticBaseEnvSettingsSource):
    """
    Source class for loading settings values from secret files.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        secrets_dir: PathType | None = None,
        case_sensitive: bool | None = None,
        env_prefix: str | None = None,
        env_ignore_empty: bool | None = None,
        env_parse_none_str: str | None = None,
        env_parse_enums: bool | None = None,
    ) -> None:
        super().__init__(
            settings_cls,
            case_sensitive,
            env_prefix,
            env_ignore_empty,
            env_parse_none_str,
            env_parse_enums,
        )
        self.secrets_dir = (
            secrets_dir if secrets_dir is not None else self.config.get("secrets_dir")
        )

    def __call__(self) -> dict[str, Any]:
        """
        Build fields from "secrets" files.
        """
        secrets: dict[str, str | None] = {}

        if self.secrets_dir is None:
            return secrets

        secrets_dirs = (
            [self.secrets_dir]
            if isinstance(self.secrets_dir, (str, os.PathLike))
            else self.secrets_dir
        )
        secrets_paths = [Path(p).expanduser() for p in secrets_dirs]
        self.secrets_paths = []

        for path in secrets_paths:
            if not path.exists():
                warnings.warn(f'directory "{path}" does not exist')
            else:
                self.secrets_paths.append(path)

        if not len(self.secrets_paths):
            return secrets

        for path in self.secrets_paths:
            if not path.is_dir():
                raise SettingsError(
                    f"secrets_dir must reference a directory, not a {path_type_label(path)}"
                )

        return super().__call__()

    @classmethod
    def find_case_path(
        cls, dir_path: Path, file_name: str, case_sensitive: bool
    ) -> Path | None:
        """
        Find a file within path's directory matching filename, optionally ignoring case.

        Args:
            dir_path: Directory path.
            file_name: File name.
            case_sensitive: Whether to search for file name case sensitively.

        Returns:
            Whether file path or `None` if file does not exist in directory.
        """
        for f in dir_path.iterdir():
            if f.name == file_name:
                return f
            elif not case_sensitive and f.name.lower() == file_name.lower():
                return f
        return None

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """
        Gets the value for field from secret file and a flag to determine whether value is complex.

        Args:
            field: The field.
            field_name: The field name.

        Returns:
            A tuple that contains the value (`None` if the file does not exist), key, and
                a flag to determine whether value is complex.
        """

        for field_key, env_name, value_is_complex in self._extract_field_info(
            field, field_name
        ):
            # paths reversed to match the last-wins behaviour of `env_file`
            for secrets_path in reversed(self.secrets_paths):
                path = self.find_case_path(secrets_path, env_name, self.case_sensitive)
                if not path:
                    # path does not exist, we currently don't return a warning for this
                    continue

                if path.is_file():
                    return path.read_text().strip(), field_key, value_is_complex
                else:
                    warnings.warn(
                        f'attempted to load secret file "{path}" but found a {path_type_label(path)} instead.',
                        stacklevel=4,
                    )

        return None, field_key, value_is_complex

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(secrets_dir={self.secrets_dir!r})"
