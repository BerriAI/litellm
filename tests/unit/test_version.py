from importlib.metadata import Distribution, PackageNotFoundError, PathDistribution
from pathlib import Path
from typing import Final

import pytest

from litellm._version import get_distribution, get_distribution_name, get_version


@pytest.mark.parametrize("name", ("litellm", "litellm-core"))
def test_distribution_version_supports_independent_installs(tmp_path: Path, name: str) -> None:
    metadata: Final = tmp_path / f"{name}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.2.3\n")
    installed: Final = PathDistribution(metadata)

    def lookup(requested: str) -> Distribution:
        if requested == name:
            return installed
        raise PackageNotFoundError(requested)

    assert get_distribution(lookup) is installed
    assert get_version(lookup) == "1.2.3"
    assert get_distribution_name(lookup) == name


def test_missing_distribution_preserves_metadata_error() -> None:
    def lookup(name: str) -> Distribution:
        raise PackageNotFoundError(name)

    assert get_version(lookup) == "unknown"
    assert get_distribution_name(lookup) == "litellm"
    with pytest.raises(PackageNotFoundError):
        get_distribution(lookup)


def test_distribution_lookup_does_not_mask_broken_metadata() -> None:
    def lookup(name: str) -> Distribution:
        raise ValueError(f"Invalid metadata: {name}")

    with pytest.raises(ValueError, match="Invalid metadata: litellm-core"):
        get_distribution(lookup)
