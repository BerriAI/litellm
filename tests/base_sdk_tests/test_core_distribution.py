from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
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


def test_core_artifacts_exclude_the_legacy_dashboard() -> None:
    source: Final = (
        METADATA
        + 'include = ["litellm/proxy/_experimental/out/**", "litellm/data.json"]\nexclude = ["**/__pycache__"]\n'
    )
    selected: Final = tomllib.loads(core_metadata(source))["tool"]["maturin"]
    assert selected["include"] == ["litellm/data.json"]
    assert selected["exclude"] == [
        "**/__pycache__",
        "litellm/proxy/_experimental/out",
        "litellm/proxy/_experimental/out/**",
    ]
    assert tomllib.loads(source)["tool"]["maturin"]["include"][0] == "litellm/proxy/_experimental/out/**"


def test_core_build_command_stages_a_buildable_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    path: Final = tmp_path / "pyproject.toml"
    path.write_text(METADATA)
    script: Final = Path(__file__).resolve().parents[2] / "scripts/prepare_core_distribution.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(path)])
    runpy.run_path(str(script), run_name="__main__")
    selected: Final = tomllib.loads(path.read_text())
    assert selected["project"]["name"] == "litellm-core"
    assert selected["project"]["optional-dependencies"]["sdk-extras"] == ["litellm-core[aws]"]


def test_http_checker_configures_offline_environment_before_sdk_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import os
    import runpy

    original_import: Final = builtins.__import__
    observed: Final[dict[str, str | None]] = {}

    def stop_at_sdk_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "litellm" or name.startswith("litellm."):
            observed.update({key: os.getenv(key) for key in ("LITELLM_LOCAL_MODEL_COST_MAP", "PYTHON_DOTENV_DISABLED")})
            raise RuntimeError("SDK initialization boundary")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setattr(builtins, "__import__", stop_at_sdk_import)
    with pytest.raises(RuntimeError, match="SDK initialization boundary"):
        runpy.run_path(str(Path(__file__).with_name("check_sdk_http.py")))
    assert observed == {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHON_DOTENV_DISABLED": "1"}
