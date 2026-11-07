"""Tests for .github/scripts/check_release_tag_version.py."""

import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "check_release_tag_version.py"
_spec: Final = importlib.util.spec_from_file_location("check_release_tag_version", _MODULE_PATH)
check_release_tag_version: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_release_tag_version
_spec.loader.exec_module(check_release_tag_version)


def _run(tmp_path: Path, tag: str, version: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    pyproject: Final = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "litellm"\nversion = "{version}"\n', encoding="utf-8")
    code: Final = check_release_tag_version.main(["check_release_tag_version.py", tag, str(pyproject)])
    captured: Final = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v1.107.0", "1.107.0"),
        ("1.107.0", "1.107.0"),
        ("v1.107.0-rc.2", "1.107.0"),
        ("1.84.0rc1", "1.84.0"),
        ("v1.84.0.dev42", "1.84.0.dev42"),
        ("v1.107.0.post1", "1.107.0.post1"),
        ("v1.83.10-stable", "1.83.10"),
        ("v1.83.3-stable.opus-4.7", "1.83.3"),
    ],
)
def test_a_tag_that_matches_the_pyproject_version_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tag: str, version: str
) -> None:
    code, out, err = _run(tmp_path, tag, version, capsys)
    assert (code, err) == (0, "")
    assert out == f"{tag} matches pyproject.toml version {version}\n"


@pytest.mark.parametrize(("tag", "version"), [("v1.107.0", "1.108.0"), ("v1.106.2", "1.107.0")])
def test_a_tag_that_does_not_match_the_pyproject_version_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tag: str, version: str
) -> None:
    code, out, err = _run(tmp_path, tag, version, capsys)
    assert code == 1
    assert out == ""
    assert err.startswith(f"::error::{tag} points at a commit whose pyproject.toml version is {version}.")


@pytest.mark.parametrize("tag", ["release-1.107.0", "v1.107", "v1.107.0foo"])
def test_a_tag_that_is_not_an_x_y_z_release_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str], tag: str) -> None:
    code, out, err = _run(tmp_path, tag, "1.107.0", capsys)
    assert code == 1
    assert out == ""
    assert err == f"::error::{tag} is not a release tag with an X.Y.Z version\n"


def test_a_malformed_pyproject_version_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(tmp_path, "v1.107.0", "1.107", capsys)
    assert code == 1
    assert out == ""
    assert err == "::error::pyproject.toml version 1.107 is not an X.Y.Z version\n"


def test_a_missing_tag_argument_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_release_tag_version.main(["check_release_tag_version.py"]) == 1
    assert capsys.readouterr().err == ("::error::usage: check_release_tag_version.py <tag> [path/to/pyproject.toml]\n")


def test_the_repo_pyproject_is_read_when_no_path_is_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    version: Final = check_release_tag_version.read_version(_REPO_ROOT / "pyproject.toml")
    assert check_release_tag_version.main(["check_release_tag_version.py", f"v{version}"]) == 0
    assert capsys.readouterr().out == f"v{version} matches pyproject.toml version {version}\n"
