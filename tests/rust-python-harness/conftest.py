from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

HARNESS_ROOT: Final = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def subprocess_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


@pytest.fixture
def cargo_project(tmp_path: Path) -> Callable[[str, str], Path]:
    def create(package: str, source: str) -> Path:
        manifest: Final = tmp_path / "Cargo.toml"
        manifest.write_text(
            f'[package]\nname = "{package}"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n'
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src/lib.rs").write_text(source)
        return manifest

    return create
