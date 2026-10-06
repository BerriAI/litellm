"""Verify file ownership and metadata from an installed SDK outside the checkout."""

import argparse
import csv
import io
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Final


def check_distribution_install(name: str) -> None:
    import litellm
    from litellm._version import get_distribution, get_version
    from litellm.rust_bridge import _native

    installed: Final = distribution(name)
    other: Final = "litellm" if name == "litellm-core" else "litellm-core"
    try:
        distribution(other)
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError(f"Unexpected overlapping distribution: {other}")
    assert get_distribution().metadata["Name"] == name
    assert get_version() == installed.version
    assert callable(litellm.completion) and callable(litellm.acompletion)
    assert _native.__file__ is not None
    record: Final = installed.read_text("RECORD")
    assert record is not None
    missing: Final = tuple(
        row[0] for row in csv.reader(io.StringIO(record)) if not installed.locate_file(row[0]).exists()
    )
    assert not missing, f"Installed distribution has missing files: {missing[:10]}"
    for command in ("litellm", "lite", "litellm-proxy"):
        suffix: Final = ".exe" if sys.platform == "win32" else ""
        assert (Path(sys.executable).parent / (command + suffix)).is_file(), f"Missing command: {command}"
    sys.stdout.write(f"{name} {installed.version}: implementation, native extension, metadata and commands intact\n")


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("litellm", "litellm-core"))
    check_distribution_install(parser.parse_args().name)
