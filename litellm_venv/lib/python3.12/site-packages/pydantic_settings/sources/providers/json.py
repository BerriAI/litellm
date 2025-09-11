"""JSON file settings source."""

from __future__ import annotations as _annotations

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from ..base import ConfigFileSourceMixin, InitSettingsSource
from ..types import DEFAULT_PATH, PathType

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings


class JsonConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a JSON file
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        json_file: PathType | None = DEFAULT_PATH,
        json_file_encoding: str | None = None,
    ):
        self.json_file_path = (
            json_file
            if json_file != DEFAULT_PATH
            else settings_cls.model_config.get("json_file")
        )
        self.json_file_encoding = (
            json_file_encoding
            if json_file_encoding is not None
            else settings_cls.model_config.get("json_file_encoding")
        )
        self.json_data = self._read_files(self.json_file_path)
        super().__init__(settings_cls, self.json_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding=self.json_file_encoding) as json_file:
            return json.load(json_file)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(json_file={self.json_file_path})"


__all__ = ["JsonConfigSettingsSource"]
