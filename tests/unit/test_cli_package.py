import base64
import csv
import hashlib
import importlib
import importlib.util
import io
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Final
from zipfile import ZipFile

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]


def _mapper() -> ModuleType:
    spec: Final = importlib.util.spec_from_file_location("cli_package", ROOT / ".github/packaging/cli_package.py")
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(directory: Path, files: dict[str, bytes]) -> Path:
    path: Final = directory / "litellm-1.0-py3-none-any.whl"
    with ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return path


def test_wheel_mapping_preserves_payload_and_records_installed_package_paths(tmp_path: Path) -> None:
    files: Final = {
        "litellm/cli/__init__.py": b"",
        "litellm/cli/main.py": b"def cli(): return 'lazy'\n",
        "litellm/__init__.py": b"raise AssertionError('SDK initialized')\n",
        "litellm-1.0.dist-info/RECORD": b"",
    }
    wheel: Final = _wheel(tmp_path, files)
    _mapper().map_wheel(wheel)
    with ZipFile(wheel) as archive:
        contents: Final = {name: archive.read(name) for name in archive.namelist()}
    assert contents["litellm_cli/main.py"] == files["litellm/cli/main.py"]
    assert contents["litellm/__init__.py"] == files["litellm/__init__.py"]
    record: Final = "litellm-1.0.dist-info/RECORD"
    rows: Final = tuple(csv.reader(io.StringIO(contents[record].decode())))
    expected: Final = {
        name: (
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
            str(len(data)),
        )
        for name, data in contents.items()
        if name != record
    }
    assert {name: (digest, size) for name, digest, size in rows} == {**expected, record: ("", "")}
    assert set(contents) == {"litellm_cli/__init__.py", "litellm_cli/main.py", "litellm/__init__.py", record}


def test_release_command_maps_each_produced_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source: Final = {"litellm/cli/main.py": b"lazy", "litellm-1.0.dist-info/RECORD": b""}
    directories: Final = (tmp_path / "first", tmp_path / "second")
    for directory in directories:
        directory.mkdir()
    wheels: Final = tuple(_wheel(directory, source) for directory in directories)
    command: Final = ROOT / ".github/packaging/cli_package.py"
    monkeypatch.setattr(sys, "argv", [str(command), *(str(wheel) for wheel in wheels)])
    runpy.run_path(str(command), run_name="__main__")
    for wheel in wheels:
        with ZipFile(wheel) as archive:
            assert archive.read("litellm_cli/main.py") == b"lazy"


def test_release_command_rejects_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    command: Final = ROOT / ".github/packaging/cli_package.py"
    monkeypatch.setattr(sys, "argv", [str(command)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(command), run_name="__main__")
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("files", "message"),
    (
        ({}, "exactly one RECORD"),
        ({"litellm-1.0.dist-info/RECORD": b""}, "does not contain the CLI package"),
        (
            {"litellm-1.0.dist-info/RECORD": b"", "litellm/cli/main.py": b"source", "litellm_cli/main.py": b"other"},
            "overwrite a wheel member",
        ),
    ),
)
def test_invalid_mapping_leaves_original_wheel_unchanged(tmp_path: Path, files: dict[str, bytes], message: str) -> None:
    wheel: Final = _wheel(tmp_path, files)
    original: Final = wheel.read_bytes()
    with pytest.raises(ValueError, match=message):
        _mapper().map_wheel(wheel)
    assert wheel.read_bytes() == original


def test_editable_cli_imports_live_source_without_initializing_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source: Final = tmp_path / "litellm/cli"
    source.mkdir(parents=True)
    (source.parent / "__init__.py").write_text("raise AssertionError('SDK initialized')\n")
    initializer: Final = source / "__init__.py"
    initializer.write_text("MARKER = 1\n")
    mapper: Final = _mapper()
    wheel: Final = _wheel(tmp_path, {"litellm-1.0.dist-info/RECORD": b""})
    mapper.map_wheel(wheel, source=source)
    with ZipFile(wheel) as archive:
        support: Final = archive.read("_litellm_cli_editable.py")
    with monkeypatch.context() as context:
        context.syspath_prepend(str(tmp_path))
        context.setattr(sys, "meta_path", list(sys.meta_path))
        context.setitem(sys.modules, "litellm_cli", None)
        context.delitem(sys.modules, "litellm_cli")
        context.delitem(sys.modules, "litellm", raising=False)
        exec(support, {})
        cli: Final = importlib.import_module("litellm_cli")
        assert cli.MARKER == 1
        assert "litellm" not in sys.modules
        initializer.write_text("MARKER = 12345\n")
        importlib.invalidate_caches()
        assert importlib.reload(cli).MARKER == 12345
        assert "litellm" not in sys.modules
