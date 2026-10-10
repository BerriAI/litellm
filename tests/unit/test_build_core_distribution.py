import sys
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT: Final = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("sdist_only", [False, True])
def test_build_command_selects_requested_distributions(tmp_path: Path, sdist_only: bool) -> None:
    from scripts.build_core_distribution import main

    output: Final = tmp_path / "dist"
    arguments: Final = ["build_core_distribution.py", "--out-dir", str(output)] + (
        ["--sdist-only"] if sdist_only else []
    )
    with (
        patch.object(sys, "argv", arguments),
        patch("scripts.build_core_distribution.stage_core_distribution") as stage,
        patch("scripts.build_core_distribution.subprocess.run") as run,
    ):
        main()
        run.assert_called_once()
        command: Final = run.call_args.args[0]
        assert command[:2] == ["uv", "build"]
        assert ("--sdist" in command) is sdist_only
        assert "--wheel" not in command
        assert command[command.index("--out-dir") + 1] == str(output)
        assert run.call_args.kwargs["check"] is True
        assert run.call_args.kwargs["cwd"] == stage.call_args.args[1]
    assert not stage.call_args.args[1].exists()


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
