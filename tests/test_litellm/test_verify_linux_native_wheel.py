from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, cast

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "verify_linux_native_wheel.py"


class _VerifierModule(Protocol):
    subprocess: ModuleType
    _load_native_module: Callable[[Path], ModuleType | None]
    main: Callable[[], int]


_SPEC: Final = importlib.util.spec_from_file_location("verify_linux_native_wheel", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOADED_VERIFIER: Final = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LOADED_VERIFIER
_SPEC.loader.exec_module(_LOADED_VERIFIER)
verifier: Final = cast(_VerifierModule, _LOADED_VERIFIER)

_EXPECTED_TAG: Final = "cp310-abi3-linux_x86_64"
_NATIVE_MEMBER: Final = "litellm/rust_bridge/_native.abi3.so"
_DIST_INFO: Final = "litellm-1.100.0.dist-info"


def _write_wheel(
    tmp_path: Path,
    *,
    filename_tag: str,
    metadata_tags: tuple[str, ...] | None = (_EXPECTED_TAG,),
    dist_info: str = _DIST_INFO,
    duplicate_wheel: bool = False,
) -> Path:
    wheel: Final = tmp_path / f"litellm-1.100.0-{filename_tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_NATIVE_MEMBER, b"synthetic native extension")
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: litellm\nVersion: 1.100.0\n",
        )
        archive.writestr(
            f"{dist_info}/RECORD",
            f"{_NATIVE_MEMBER},,\n{dist_info}/WHEEL,,\n",
        )
        if metadata_tags is not None:
            wheel_metadata: Final = (
                "Wheel-Version: 1.0\nGenerator: regression-test\nRoot-Is-Purelib: false\n"
                + "".join(f"Tag: {tag}\n" for tag in metadata_tags)
            )
            archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
            if duplicate_wheel:
                archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
    return wheel


def _fake_subprocess_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
    if command == ("rustc", "--version"):
        stdout = "rustc 1.98.0 (regression-test)\n"
    elif "--sections" in command:
        stdout = "[ 1] .text PROGBITS\n"
    elif "--dyn-syms" in command:
        stdout = "PyInit__native\n"
    else:
        raise AssertionError(f"unexpected subprocess command: {command}")
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _run_verifier(
    monkeypatch: pytest.MonkeyPatch,
    wheel: Path,
    *,
    exposes_panic: bool = False,
) -> int:
    native_module: Final = ModuleType("litellm.rust_bridge._native")
    if exposes_panic:
        setattr(native_module, "_panic_for_test", lambda: None)

    def _fake_load_native_module(_: Path) -> ModuleType:
        return native_module

    monkeypatch.setattr(verifier, "_load_native_module", _fake_load_native_module)
    monkeypatch.setattr(verifier.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(sys, "argv", [str(_MODULE_PATH), str(wheel)])
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(wheel.parent / "summary.md"))
    return verifier.main()


def test_accepts_expected_release_wheel_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(monkeypatch, wheel) == 0


def test_rejects_cp312_version_specific_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tag: Final = "cp312-cp312-linux_x86_64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    assert _run_verifier(monkeypatch, wheel) == 1


def test_rejects_non_linux_platform_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tag: Final = "cp310-abi3-win_amd64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    assert _run_verifier(monkeypatch, wheel) == 1


@pytest.mark.parametrize(
    "metadata_tags",
    [None, ("cp312-cp312-linux_x86_64",)],
    ids=["missing", "mismatched"],
)
def test_rejects_missing_or_mismatched_wheel_metadata_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_tags: tuple[str, ...] | None,
) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG, metadata_tags=metadata_tags)

    assert _run_verifier(monkeypatch, wheel) == 1


def test_rejects_wheel_metadata_from_wrong_dist_info_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        dist_info="decoy-1.0.0.dist-info",
    )

    assert _run_verifier(monkeypatch, wheel) == 1


def test_rejects_duplicate_wheel_metadata_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        metadata_tags=(_EXPECTED_TAG, _EXPECTED_TAG),
    )

    assert _run_verifier(monkeypatch, wheel) == 1


def test_rejects_duplicate_wheel_metadata_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        wheel: Final = _write_wheel(
            tmp_path,
            filename_tag=_EXPECTED_TAG,
            duplicate_wheel=True,
        )

    assert _run_verifier(monkeypatch, wheel) == 1


def test_rejects_production_module_exposing_panic_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(monkeypatch, wheel, exposes_panic=True) == 1
