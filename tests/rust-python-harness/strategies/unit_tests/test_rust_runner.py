from __future__ import annotations

import shutil
from pathlib import Path
from typing import Final

import pytest

from .rust_runner import run_rust_tests


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for native runner integration")
def test_collects_and_runs_native_tests_and_propagates_failure(tmp_path: Path) -> None:
    manifest: Final = tmp_path / "Cargo.toml"
    manifest.write_text('[package]\nname = "harness-runner-check"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n')
    (tmp_path / "src").mkdir()
    source: Final = tmp_path / "src/lib.rs"
    source.write_text("#[test] fn test_parity() { assert_eq!(2 + 2, 4); }\n")
    inventory: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity", collect_only=True)
    assert inventory.exit_code == 0, inventory.output
    assert inventory.tests == ("test_parity",)
    passing: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity")
    assert passing.exit_code == 0, passing.output
    source.write_text("#[test] fn test_parity() { assert_eq!(2 + 2, 5); }\n")
    failed: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity")
    assert failed.exit_code != 0
    assert "test_parity" in failed.output
