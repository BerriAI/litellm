import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest


@pytest.fixture(scope="module")
def detector() -> ModuleType:
    path: Final = Path(__file__).resolve().parents[1] / "code_coverage_tests" / "recursive_detector.py"
    spec: Final = importlib.util.spec_from_file_location("recursive_detector", path)
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source_dir",
    ("cli", "deploy", "enterprise", "packaging", "proxy-extras", "runtime", "rust", "scripts", "ui", "vscode-extension"),
)
def test_workspace_sources_do_not_hide_sdk_recursion(detector: ModuleType, tmp_path: Path, source_dir: str) -> None:
    workspace: Final = tmp_path / source_dir
    workspace.mkdir()
    (workspace / "helper.py").write_text("def cycle(): return cycle()")
    sdk: Final = tmp_path / "sdk.py"
    sdk.write_text("def cycle(): return cycle()")
    assert detector.find_recursive_functions_in_directory(tmp_path) == ({str(sdk): ["cycle"]}, {})


def test_nested_sdk_directories_remain_scanned(detector: ModuleType, tmp_path: Path) -> None:
    nested: Final = tmp_path / "provider" / "scripts"
    nested.mkdir(parents=True)
    sdk: Final = nested / "source.py"
    sdk.write_text("def cycle(): return cycle()")
    assert detector.find_recursive_functions_in_directory(str(tmp_path)) == ({str(sdk): ["cycle"]}, {})


def test_empty_sdk_has_no_recursion(detector: ModuleType, tmp_path: Path) -> None:
    assert detector.find_recursive_functions_in_directory(str(tmp_path)) == ({}, {})


def test_workspace_python_is_not_parsed(detector: ModuleType, tmp_path: Path) -> None:
    workspace: Final = tmp_path / "scripts"
    workspace.mkdir()
    (workspace / "helper.py").write_text("def unfinished(")
    assert detector.find_recursive_functions_in_directory(str(tmp_path)) == ({}, {})


def test_invalid_sdk_python_still_fails(detector: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "sdk.py").write_text("def unfinished(")
    with pytest.raises(SyntaxError):
        detector.find_recursive_functions_in_directory(str(tmp_path))
