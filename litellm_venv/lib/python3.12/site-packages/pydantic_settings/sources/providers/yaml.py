"""YAML file settings source."""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from ..base import ConfigFileSourceMixin, InitSettingsSource
from ..types import DEFAULT_PATH, PathType

if TYPE_CHECKING:
    import yaml

    from pydantic_settings.main import BaseSettings
else:
    yaml = None


def import_yaml() -> None:
    global yaml
    if yaml is not None:
        return
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is not installed, run `pip install pydantic-settings[yaml]`"
        ) from e


class YamlConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a yaml file
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: PathType | None = DEFAULT_PATH,
        yaml_file_encoding: str | None = None,
        yaml_config_section: str | None = None,
    ):
        self.yaml_file_path = (
            yaml_file
            if yaml_file != DEFAULT_PATH
            else settings_cls.model_config.get("yaml_file")
        )
        self.yaml_file_encoding = (
            yaml_file_encoding
            if yaml_file_encoding is not None
            else settings_cls.model_config.get("yaml_file_encoding")
        )
        self.yaml_config_section = (
            yaml_config_section
            if yaml_config_section is not None
            else settings_cls.model_config.get("yaml_config_section")
        )
        self.yaml_data = self._read_files(self.yaml_file_path)

        if self.yaml_config_section:
            try:
                self.yaml_data = self.yaml_data[self.yaml_config_section]
            except KeyError:
                raise KeyError(
                    f'yaml_config_section key "{self.yaml_config_section}" not found in {self.yaml_file_path}'
                )
        super().__init__(settings_cls, self.yaml_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        import_yaml()
        with open(file_path, encoding=self.yaml_file_encoding) as yaml_file:
            return yaml.safe_load(yaml_file) or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(yaml_file={self.yaml_file_path})"


__all__ = ["YamlConfigSettingsSource"]
