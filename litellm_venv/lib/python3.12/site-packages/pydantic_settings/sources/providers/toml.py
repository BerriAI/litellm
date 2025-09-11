"""TOML file settings source."""

from __future__ import annotations as _annotations

import sys
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from ..base import ConfigFileSourceMixin, InitSettingsSource
from ..types import DEFAULT_PATH, PathType

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = None
    import tomli
else:
    tomllib = None
    tomli = None


def import_toml() -> None:
    global tomli
    global tomllib
    if sys.version_info < (3, 11):
        if tomli is not None:
            return
        try:
            import tomli
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "tomli is not installed, run `pip install pydantic-settings[toml]`"
            ) from e
    else:
        if tomllib is not None:
            return
        import tomllib


class TomlConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a TOML file
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        toml_file: PathType | None = DEFAULT_PATH,
    ):
        self.toml_file_path = (
            toml_file
            if toml_file != DEFAULT_PATH
            else settings_cls.model_config.get("toml_file")
        )
        self.toml_data = self._read_files(self.toml_file_path)
        super().__init__(settings_cls, self.toml_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        import_toml()
        with open(file_path, mode="rb") as toml_file:
            if sys.version_info < (3, 11):
                return tomli.load(toml_file)
            return tomllib.load(toml_file)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(toml_file={self.toml_file_path})"
