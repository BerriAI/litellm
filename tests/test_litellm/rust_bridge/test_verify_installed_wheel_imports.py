from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, cast


class _VerifierModule(Protocol):
    def provenance_failures(
        self,
        *,
        prefix: Path,
        checkout: Path,
        module_files: Mapping[str, Path],
        direct_url: str | None,
        native_available: bool,
    ) -> tuple[str, ...]: ...


_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "verify_installed_wheel_imports.py"
_SPEC: Final = importlib.util.spec_from_file_location("verify_installed_wheel_imports", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOADED_VERIFIER: Final = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LOADED_VERIFIER
_SPEC.loader.exec_module(_LOADED_VERIFIER)
verifier: Final = cast(_VerifierModule, _LOADED_VERIFIER)


def _failures(
    *,
    prefix: Path,
    checkout: Path,
    module_files: Mapping[str, Path],
    direct_url: str | None = '{"dir_info": {"editable": false}}',
    native_available: bool = True,
) -> tuple[str, ...]:
    return verifier.provenance_failures(
        prefix=prefix,
        checkout=checkout,
        module_files=module_files,
        direct_url=direct_url,
        native_available=native_available,
    )


def test_accepts_installed_non_editable_modules(tmp_path: Path) -> None:
    prefix: Final = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    checkout: Final = tmp_path / "checkout"
    module_files: Final = {
        "litellm": prefix / "litellm" / "__init__.py",
        "litellm.rust_bridge._native": prefix / "litellm" / "rust_bridge" / "_native.abi3.so",
    }

    assert _failures(prefix=prefix, checkout=checkout, module_files=module_files) == ()


def test_rejects_module_under_checkout(tmp_path: Path) -> None:
    prefix: Final = tmp_path
    checkout: Final = tmp_path / "checkout"
    module_files: Final = {"litellm": checkout / "litellm" / "__init__.py"}

    assert _failures(prefix=prefix, checkout=checkout, module_files=module_files) == (
        f"litellm is under the checkout: {(checkout / 'litellm' / '__init__.py').resolve()}",
    )


def test_rejects_editable_direct_url(tmp_path: Path) -> None:
    prefix: Final = tmp_path / "site-packages"
    checkout: Final = tmp_path / "checkout"
    module_files: Final = {"litellm": prefix / "litellm" / "__init__.py"}

    assert _failures(
        prefix=prefix,
        checkout=checkout,
        module_files=module_files,
        direct_url='{"dir_info": {"editable": true}}',
    ) == ("litellm distribution is editable",)


def test_rejects_unavailable_native_bridge(tmp_path: Path) -> None:
    prefix: Final = tmp_path / "site-packages"
    checkout: Final = tmp_path / "checkout"
    module_files: Final = {"litellm": prefix / "litellm" / "__init__.py"}

    assert _failures(
        prefix=prefix,
        checkout=checkout,
        module_files=module_files,
        native_available=False,
    ) == ("litellm.rust_bridge.native_bridge_available() is false",)
