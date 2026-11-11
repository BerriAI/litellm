from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from .rust_runner import RustTarget, RustTestScope, enumerate_rust_tests, run_command, run_rust_tests


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for native runner integration")
def test_collects_and_runs_native_tests_and_propagates_failure(
    tmp_path: Path,
    cargo_project: Callable[[str, str], Path],
) -> None:
    manifest: Final = cargo_project("harness-runner-check", "#[test] fn test_parity() { assert_eq!(2 + 2, 4); }\n")
    source: Final = tmp_path / "src/lib.rs"
    inventory: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity", collect_only=True)
    assert inventory.exit_code == 0, inventory.output
    assert inventory.tests == ("test_parity",)
    passing: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity")
    assert passing.exit_code == 0, passing.output
    source.write_text("#[test] fn test_parity() { assert_eq!(2 + 2, 5); }\n")
    failed: Final = run_rust_tests(manifest, "harness-runner-check", "test_parity")
    assert failed.exit_code != 0
    assert "test_parity" in failed.output


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for compiled inventory tests")
def test_discovers_compiled_fully_qualified_tests(tmp_path: Path) -> None:
    workspace: Final = tmp_path / "litellm-rust"
    source: Final = workspace / "src"
    external: Final = source / "ocr" / "external.rs"
    external.parent.mkdir(parents=True)
    (workspace / "Cargo.toml").write_text(
        '[package]\nname = "inventory-fixture"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n',
        encoding="utf-8",
    )
    (source / "lib.rs").write_text(
        "#[cfg(test)]\n"
        "mod ocr {\n"
        "    mod external;\n"
        "    #[test] fn same_name() {}\n"
        "    #[cfg(any())] #[test] fn compiled_out() {}\n"
        "    macro_rules! generate_test { ($name:ident) => { #[test] fn $name() {} }; }\n"
        "    generate_test!(generated_case);\n"
        "}\n",
        encoding="utf-8",
    )
    external.write_text("#[test] fn same_name() {}\n", encoding="utf-8")
    run_command(("cargo", "generate-lockfile", "--offline"), workspace)
    target: Final = RustTarget(package="inventory-fixture", name="inventory_fixture", kind="lib")
    scope: Final = RustTestScope(target=target, modules=("ocr",))

    inventory: Final = enumerate_rust_tests(tmp_path, (scope,))

    assert frozenset(identity.name for identity in inventory) == frozenset(
        ("ocr::same_name", "ocr::external::same_name", "ocr::generated_case")
    )
