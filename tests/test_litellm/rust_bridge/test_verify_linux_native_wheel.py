from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final, Protocol, cast

import pytest


class _CommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]: ...


class _VerifierModule(Protocol):
    main: Callable[
        [
            Sequence[str] | None,
            Mapping[str, str] | None,
            Callable[[Path], ModuleType | None],
            _CommandRunner,
        ],
        int,
    ]


_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "verify_linux_native_wheel.py"
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


def test_accepts_expected_release_wheel_tags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(wheel) == 0
    captured: Final = capsys.readouterr()
    summary: Final = (tmp_path / "summary.md").read_text()
    assert captured.err == ""
    assert "| Validation | Expected | Result |" in summary
    assert "| Yes | X |" not in summary


@dataclass(frozen=True, slots=True)
class _InvalidWheelCase:
    filename_tag: str = _EXPECTED_TAG
    metadata_tags: tuple[str, ...] | None = (_EXPECTED_TAG,)
    dist_info: str = _DIST_INFO
    duplicate_wheel: bool = False
    exposes_panic: bool = False
    expected_error: str = ""


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            _InvalidWheelCase(
                filename_tag="cp312-cp312-linux_x86_64",
                metadata_tags=("cp312-cp312-linux_x86_64",),
                expected_error="unexpected Python tag: expected cp310, found cp312",
            ),
            id="version-specific-python",
        ),
        pytest.param(
            _InvalidWheelCase(
                filename_tag="cp310-abi3-win_amd64",
                metadata_tags=("cp310-abi3-win_amd64",),
                expected_error="unexpected platform tag: expected linux_x86_64, found win_amd64",
            ),
            id="non-linux-platform",
        ),
        pytest.param(
            _InvalidWheelCase(
                metadata_tags=None,
                expected_error="required dist-info file counts are invalid",
            ),
            id="metadata-tag-missing",
        ),
        pytest.param(
            _InvalidWheelCase(
                metadata_tags=("cp312-cp312-linux_x86_64",),
                expected_error=(
                    "WHEEL tags do not match filename: expected cp310-abi3-linux_x86_64, found cp312-cp312-linux_x86_64"
                ),
            ),
            id="metadata-tag-mismatched",
        ),
        pytest.param(
            _InvalidWheelCase(
                dist_info="decoy-1.0.0.dist-info",
                expected_error="unexpected dist-info directories: expected litellm-1.100.0.dist-info",
            ),
            id="wrong-dist-info-directory",
        ),
        pytest.param(
            _InvalidWheelCase(
                metadata_tags=(_EXPECTED_TAG, _EXPECTED_TAG),
                expected_error=(
                    "WHEEL tags do not match filename: expected cp310-abi3-linux_x86_64, "
                    "found cp310-abi3-linux_x86_64, cp310-abi3-linux_x86_64"
                ),
            ),
            id="duplicate-metadata-tag",
        ),
        pytest.param(
            _InvalidWheelCase(
                duplicate_wheel=True,
                expected_error="required dist-info file counts are invalid",
            ),
            id="duplicate-metadata-file",
        ),
        pytest.param(
            _InvalidWheelCase(
                exposes_panic=True,
                expected_error="production native module exposes _panic_for_test",
            ),
            id="panic-hook-exposed",
        ),
    ),
)
def test_rejects_invalid_wheel_with_specific_diagnostic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: _InvalidWheelCase,
) -> None:
    if case.duplicate_wheel:
        with pytest.warns(UserWarning, match="Duplicate name"):
            wheel: Final = _write_wheel(
                tmp_path,
                filename_tag=case.filename_tag,
                metadata_tags=case.metadata_tags,
                dist_info=case.dist_info,
                duplicate_wheel=True,
            )
    else:
        wheel = _write_wheel(
            tmp_path,
            filename_tag=case.filename_tag,
            metadata_tags=case.metadata_tags,
            dist_info=case.dist_info,
        )

    assert _run_verifier(wheel, exposes_panic=case.exposes_panic) == 1
    captured: Final = capsys.readouterr()
    summary: Final = (tmp_path / "summary.md").read_text()
    assert case.expected_error in captured.err
    assert "| Yes | X |" in summary
