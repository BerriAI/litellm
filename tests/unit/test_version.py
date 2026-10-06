import runpy
from pathlib import Path
from typing import Final
from unittest.mock import patch

import importlib_metadata
import pytest


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ({"litellm": "1.2.3"}, "1.2.3"),
        ({"litellm-core": "2.3.4"}, "2.3.4"),
        ({}, "unknown"),
    ],
)
def test_version_uses_installed_distribution(installed: dict[str, str], expected: str) -> None:
    def lookup(name: str) -> str:
        if name not in installed:
            raise importlib_metadata.PackageNotFoundError(name)
        return installed[name]

    with patch("importlib_metadata.version", side_effect=lookup):
        result: Final = runpy.run_path(str(Path(__file__).resolve().parents[2] / "litellm/_version.py"))
    assert result["version"] == expected
