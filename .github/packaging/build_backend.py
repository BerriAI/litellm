import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import maturin
from cli_package import PACKAGE_DIR, map_wheel
from maturin import (
    build_sdist as build_sdist,
)
from maturin import (
    get_requires_for_build_editable as get_requires_for_build_editable,
)
from maturin import (
    get_requires_for_build_sdist as get_requires_for_build_sdist,
)
from maturin import (
    get_requires_for_build_wheel as get_requires_for_build_wheel,
)
from maturin import (
    prepare_metadata_for_build_editable as prepare_metadata_for_build_editable,
)
from maturin import (
    prepare_metadata_for_build_wheel as prepare_metadata_for_build_wheel,
)


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    filename: Final = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
    map_wheel(Path(wheel_directory) / filename)
    return filename


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    filename: Final = maturin.build_editable(wheel_directory, config_settings, metadata_directory)
    map_wheel(Path(wheel_directory) / filename, source=Path(__file__).resolve().parents[2] / PACKAGE_DIR)
    return filename


if __name__ == "__main__":
    build_editable(sys.argv[1])
