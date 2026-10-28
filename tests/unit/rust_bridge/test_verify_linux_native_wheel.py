from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
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
    native_bytes: bytes = b"synthetic native extension",
    kind: str = "extension",
) -> Path:
    distribution: Final = "litellm_gateway" if kind == "gateway" else "litellm"
    metadata_directory: Final = f"{distribution}-1.100.0.dist-info" if kind == "gateway" else dist_info
    native_member: Final = (
        f"{distribution}-1.100.0.data/scripts/litellm-gateway" if kind == "gateway" else _NATIVE_MEMBER
    )
    wheel: Final = tmp_path / f"{distribution}-1.100.0-{filename_tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(native_member, native_bytes)
        archive.writestr(
            f"{metadata_directory}/METADATA",
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.100.0\n",
        )
        archive.writestr(
            f"{metadata_directory}/RECORD",
            f"{native_member},,\n{metadata_directory}/WHEEL,,\n",
        )
        if metadata_tags is not None:
            wheel_metadata: Final = (
                "Wheel-Version: 1.0\nGenerator: regression-test\nRoot-Is-Purelib: false\n"
                + "".join(f"Tag: {tag}\n" for tag in metadata_tags)
            )
            archive.writestr(f"{metadata_directory}/WHEEL", wheel_metadata)
            if duplicate_wheel:
                archive.writestr(f"{metadata_directory}/WHEEL", wheel_metadata)
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
    kind: str = "extension",
    loads: bool = True,
    run_command: _CommandRunner = _fake_subprocess_run,
) -> int:
    native_module: Final = (
        _NativeModuleWithPanicHook("litellm.rust_bridge._native")
        if exposes_panic
        else ModuleType("litellm.rust_bridge._native")
    )

    def _fake_load_native_module(_: Path) -> ModuleType | None:
        assert kind != "gateway", "A standalone binary must not be imported as a Python extension"
        return native_module if loads else None

    environment: Final = MappingProxyType({"GITHUB_STEP_SUMMARY": str(wheel.parent / "summary.md")})
    return verifier.main(
        (str(_MODULE_PATH), kind, str(wheel)),
        environment,
        _fake_load_native_module,
        run_command,
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


@pytest.mark.parametrize("embedded", (False, True))
def test_vocabulary_is_packaged_once(tmp_path: Path, embedded: bool) -> None:
    ranks: Final = b"AA== 0\nAQ== 1\nAg== 2\n"
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        native_bytes=b"native engine" + (ranks if embedded else b""),
    )
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("litellm/litellm_core_utils/tokenizers/" + "a" * 40, ranks)

    assert _run_verifier(wheel) == (1 if embedded else 0)


@pytest.mark.parametrize("kind, limit", (("extension", 48_000_000), ("gateway", 37_000_000)))
@pytest.mark.parametrize("excess", (0, 1), ids=("at-limit", "over-limit"))
def test_enforces_uncompressed_native_size_limit(tmp_path: Path, kind: str, limit: int, excess: int) -> None:
    tag: Final = "py3-none-linux_x86_64" if kind == "gateway" else _EXPECTED_TAG
    wheel: Final = _write_wheel(
        tmp_path, filename_tag=tag, metadata_tags=(tag,), native_bytes=b"x" * (limit + excess), kind=kind
    )

    assert _run_verifier(wheel, kind=kind) == (0 if excess == 0 else 1)
    summary: Final = (tmp_path / "summary.md").read_text()
    label: Final = "Gateway binary" if kind == "gateway" else "Native extension"
    assert f"| {label} | {(limit + excess) / 1_000_000:.2f} MB |" in summary


def test_keeps_both_artifact_reports_in_summary(tmp_path: Path) -> None:
    extension: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)
    gateway: Final = _write_wheel(
        tmp_path, filename_tag="py3-none-linux_x86_64", metadata_tags=("py3-none-linux_x86_64",), kind="gateway"
    )

    assert _run_verifier(extension) == 0
    assert _run_verifier(gateway, kind="gateway") == 0
    summary: Final = (tmp_path / "summary.md").read_text()
    assert "## Native wheel build report" in summary
    assert "## Gateway wheel build report" in summary


@pytest.mark.parametrize("kind, other", (("extension", "gateway"), ("gateway", "extension")))
def test_rejects_wheel_under_wrong_profile(tmp_path: Path, kind: str, other: str) -> None:
    tag: Final = "py3-none-linux_x86_64" if kind == "gateway" else _EXPECTED_TAG
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,), kind=kind)

    assert _run_verifier(wheel, kind=other) == 1


def test_rejects_unknown_profile(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(wheel, kind="unknown") == 2


@pytest.mark.parametrize("arguments", ((), ("extension",), ("extension", "wheel.whl", "extra")))
def test_rejects_wrong_argument_count(arguments: tuple[str, ...]) -> None:
    assert verifier.main((str(_MODULE_PATH), *arguments), None, lambda _: None, _fake_subprocess_run) == 2


def test_load_failure_does_not_report_a_panic_hook_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    assert _run_verifier(wheel, loads=False) == 1
    summary: Final = (tmp_path / "summary.md").read_text()
    assert summary.count("| X |") == 1
    assert "| Native module loads | Yes | X |" in summary
    assert "native module did not load" in capsys.readouterr().err


def test_preserves_size_report_when_readelf_fails(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    def _failing_readelf(
        command: tuple[str, ...], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "readelf":
            raise subprocess.CalledProcessError(1, command)
        return _fake_subprocess_run(command, check=check, capture_output=capture_output, text=text)

    with pytest.raises(subprocess.CalledProcessError):
        _run_verifier(wheel, run_command=_failing_readelf)

    summary: Final = (tmp_path / "summary.md").read_text()
    assert "## Native wheel build report" in summary
    assert "| Compressed wheel |" in summary
