"""Tests for .github/scripts/read_rc_version.py."""

import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "read_rc_version.py"
_spec: Final = importlib.util.spec_from_file_location("read_rc_version", _MODULE_PATH)
read_rc_version: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = read_rc_version
_spec.loader.exec_module(read_rc_version)


def _run(tmp_path, version, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "litellm"\nversion = "{version}"\n', encoding="utf-8")
    code = read_rc_version.main(["read_rc_version.py", str(pyproject)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("version", ["1.104.0", "2.0.0", "10.250.0"])
def test_an_x_y_0_version_is_printed_as_a_github_output_line(tmp_path, capsys, version):
    code, out, err = _run(tmp_path, version, capsys)
    assert (code, out, err) == (0, f"version={version}\n", "")


@pytest.mark.parametrize("version", ["1.104.1", "1.104.0rc1", "1.104", "v1.104.0", "1.104.0.dev1"])
def test_a_non_release_version_exits_1_without_printing_a_version(tmp_path, capsys, version):
    code, out, err = _run(tmp_path, version, capsys)
    assert code == 1
    assert out == ""
    assert err == f"::error::pyproject.toml version {version} is not an X.Y.0 release version\n"


def test_the_repo_pyproject_is_read_when_no_path_is_given(monkeypatch, capsys):
    monkeypatch.chdir(_REPO_ROOT)
    assert read_rc_version.main(["read_rc_version.py"]) == 0
    assert capsys.readouterr().out == f"version={read_rc_version.read_version(_REPO_ROOT / 'pyproject.toml')}\n"
