from pathlib import Path
from typing import Final

import pytest
import tomllib

from scripts.prepare_core_distribution import core_metadata, prepare_core_distribution

METADATA: Final = """[project]
name = "litellm"
version = "1.2.3"
dependencies = [
    "legacy-feature>=1",
]
[project.optional-dependencies]
sdk-extras = ["litellm[aws]"]
semantic-router = ["semantic-router>=0.1"]
[tool.litellm.core]
dependencies = ["httpx>=0.28"]
legacy-only-extras = ["semantic-router"]
[tool.maturin]
module-name = "litellm.rust_bridge._native"
"""


def test_core_build_selects_dependencies_without_changing_import_package(tmp_path: Path) -> None:
    path: Final = tmp_path / "pyproject.toml"
    path.write_text(METADATA)
    prepare_core_distribution(path)
    result: Final = tomllib.loads(path.read_text())
    assert result["project"]["name"] == "litellm-core"
    assert result["project"]["version"] == "1.2.3"
    assert result["project"]["dependencies"] == ["httpx>=0.28"]
    assert result["project"]["optional-dependencies"]["sdk-extras"] == ["litellm-core[aws]"]
    assert "semantic-router" not in result["project"]["optional-dependencies"]
    assert result["tool"]["maturin"]["module-name"] == "litellm.rust_bridge._native"


@pytest.mark.parametrize(
    "source,reason",
    (
        (METADATA.replace('name = "litellm"', 'name = "other"'), "unmodified litellm"),
        (METADATA.replace('["httpx>=0.28"]', "[1]"), "explicit list"),
        (METADATA.replace('["httpx>=0.28"]', '"httpx"'), "explicit list"),
        (METADATA.replace("[project]\n", "[project] # inline\n"), "Missing project"),
        (METADATA.replace('dependencies = [\n    "legacy-feature>=1",\n]', "dependencies = []"), "multiline"),
    ),
)
def test_invalid_core_metadata_fails_before_writing(tmp_path: Path, source: str, reason: str) -> None:
    path: Final = tmp_path / "pyproject.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match=reason):
        prepare_core_distribution(path)
    assert path.read_text() == source


def test_preparing_core_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="unmodified litellm"):
        core_metadata(core_metadata(METADATA))
