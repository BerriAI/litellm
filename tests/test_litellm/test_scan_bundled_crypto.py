"""
Behavior checks on docker/scan_bundled_crypto.sh, the build-time ELF gate of
the FIPS image. The gate is exercised for real over a synthetic site-packages
tree, so a waiver that lets a bundled OpenSSL through fails here.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

SCRIPT_PATH: Final = Path(__file__).resolve().parents[2] / "docker" / "scan_bundled_crypto.sh"
ELF_MAGIC: Final = b"\x7fELF"

pytestmark = pytest.mark.skipif(
    not SCRIPT_PATH.exists() or shutil.which("objdump") is None,
    reason="scan_bundled_crypto.sh or objdump not present",
)


def _write_elf_with_bundled_openssl(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    path.write_bytes(ELF_MAGIC + b"\0" * 60 + b"OpenSSL 3.0.15 3 Sep 2024\0")
    return path


def _run_scan(scan_root: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    inventory: Final = tmp_path / "inventory.txt"
    inventory.write_text("")
    return subprocess.run(
        ["sh", str(SCRIPT_PATH), str(scan_root)],
        env={
            **os.environ,
            "SCAN_INVENTORY": str(inventory),
            "SCAN_REPORT": str(tmp_path / "report.txt"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_cryptography_bundling_its_own_openssl_fails_the_scan(tmp_path: Path) -> None:
    binding: Final = _write_elf_with_bundled_openssl(
        tmp_path / "venv" / "site-packages" / "cryptography" / "hazmat" / "bindings" / "_rust.abi3.so"
    )

    result: Final = _run_scan(tmp_path / "venv", tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"VIOLATION {binding} [bundled-OpenSSL]" in result.stdout, result.stdout
    assert "WAIVED" not in result.stdout, result.stdout


def test_unrelated_waived_binary_still_passes_the_scan(tmp_path: Path) -> None:
    _write_elf_with_bundled_openssl(tmp_path / "venv" / "site-packages" / "grpc" / "_cython" / "cygrpc.abi3.so")

    result: Final = _run_scan(tmp_path / "venv", tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WAIVED" in result.stdout and "cygrpc" in result.stdout, result.stdout
