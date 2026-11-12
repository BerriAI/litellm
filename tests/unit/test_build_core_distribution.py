import sys
from pathlib import Path
from typing import Final

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT: Final = Path(__file__).resolve().parents[2]


def test_core_manifest_preserves_runtime_dependencies_without_extras() -> None:
    manifest: Final = ROOT / "packaging/litellm-core/pyproject.toml"
    assert manifest.is_file(), "The core distribution needs its own build manifest"
    core: Final = tomllib.loads(manifest.read_text())
    legacy: Final = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert core["project"]["name"] == "litellm-core"
    removed: Final = {"boto3", "tokenizers", "huggingface-hub"}
    core_dependencies: Final = {Requirement(value).name for value in core["project"]["dependencies"]}
    legacy_dependencies: Final = {Requirement(value).name for value in legacy["project"]["dependencies"]}
    assert not core_dependencies & removed
    assert removed <= legacy_dependencies
    assert "jsonschema" in core_dependencies
    assert legacy_dependencies - removed <= core_dependencies
    core_requirements: Final = {Requirement(value) for value in core["project"]["dependencies"]}
    retained_requirements: Final = {
        Requirement(value) for value in legacy["project"]["dependencies"] if Requirement(value).name not in removed
    }
    assert retained_requirements <= core_requirements
    assert not core["project"].get("optional-dependencies")
    assert not core["project"].get("scripts")
