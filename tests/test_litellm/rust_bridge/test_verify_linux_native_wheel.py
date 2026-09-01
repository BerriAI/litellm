from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final

import pytest

from litellm.rust_bridge import verify_linux_native_wheel as verifier

_MODULE_PATH: Final = Path(verifier.__file__)

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


def _fake_subprocess_run(
    command: tuple[str, ...],
    *,
    check: bool,
    capture_output: bool,
    text: bool,
) -> subprocess.CompletedProcess[str]:
    assert check and capture_output and text
    if command == ("rustc", "--version"):
        return subprocess.CompletedProcess(command, 0, stdout="rustc 1.98.0 (regression-test)\n", stderr="")
    if "--sections" in command:
        return subprocess.CompletedProcess(command, 0, stdout="[ 1] .text PROGBITS\n", stderr="")
    if "--dyn-syms" in command:
        return subprocess.CompletedProcess(command, 0, stdout="PyInit__native\n", stderr="")
    raise AssertionError(f"unexpected subprocess command: {command}")


class _NativeModuleWithPanicHook(ModuleType):
    def _panic_for_test(self) -> None:
        return None


def _run_verifier(
    wheel: Path,
    *,
    exposes_panic: bool = False,
) -> int:
    native_module: Final = (
        _NativeModuleWithPanicHook("litellm.rust_bridge._native")
        if exposes_panic
        else ModuleType("litellm.rust_bridge._native")
    )

    def _fake_load_native_module(_: Path) -> ModuleType:
        return native_module

    environment: Final = MappingProxyType({"GITHUB_STEP_SUMMARY": str(wheel.parent / "summary.md")})
    return verifier.main(
        (str(_MODULE_PATH), str(wheel)),
        environment,
        _fake_load_native_module,
        _fake_subprocess_run,
    )


def test_accepts_expected_release_wheel_tags(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(wheel) == 0


def test_rejects_cp312_version_specific_wheel(tmp_path: Path) -> None:
    tag: Final = "cp312-cp312-linux_x86_64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    assert _run_verifier(wheel) == 1


def test_rejects_non_linux_platform_tag(tmp_path: Path) -> None:
    tag: Final = "cp310-abi3-win_amd64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    assert _run_verifier(wheel) == 1


@pytest.mark.parametrize(
    "metadata_tags",
    (None, ("cp312-cp312-linux_x86_64",)),
    ids=("missing", "mismatched"),
)
def test_rejects_missing_or_mismatched_wheel_metadata_tag(
    tmp_path: Path,
    metadata_tags: tuple[str, ...] | None,
) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG, metadata_tags=metadata_tags)

    assert _run_verifier(wheel) == 1


def test_rejects_wheel_metadata_from_wrong_dist_info_directory(
    tmp_path: Path,
) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        dist_info="decoy-1.0.0.dist-info",
    )

    assert _run_verifier(wheel) == 1


def test_rejects_duplicate_wheel_metadata_tags(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        metadata_tags=(_EXPECTED_TAG, _EXPECTED_TAG),
    )

    assert _run_verifier(wheel) == 1


def test_rejects_duplicate_wheel_metadata_file(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        wheel: Final = _write_wheel(
            tmp_path,
            filename_tag=_EXPECTED_TAG,
            duplicate_wheel=True,
        )

    assert _run_verifier(wheel) == 1


def test_rejects_production_module_exposing_panic_hook(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(wheel, exposes_panic=True) == 1
