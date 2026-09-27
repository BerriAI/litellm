"""The FIPS image bundled-crypto gate fails unreviewed or bundled crypto and passes a system-OpenSSL build."""

import _ssl
import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SCANNER: Final = REPO_ROOT / "docker" / "scan_bundled_crypto.sh"
SOURCE_INVENTORY: Final = REPO_ROOT / "docker" / "fips_elf_inventory.txt"
CYGRPC_INVENTORY_PATH: Final = (
    "app/.venv/lib/python3.13/site-packages/grpc/_cython/cygrpc.cpython-313-x86_64-linux-gnu.so"
)
CYGRPC_SO: Final = Path(str(importlib.import_module("grpc._cython.cygrpc").__file__))


def _inventory_for(root: Path) -> Path:
    entries: Final = tuple(f"{root}{line}" for line in SOURCE_INVENTORY.read_text().splitlines() if line.strip())
    inventory: Final = root.parent / "fips_elf_inventory.txt"
    inventory.write_text("\n".join(entries) + "\n")
    return inventory


def _run_scan(root: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(SCANNER), str(root)],
        cwd=REPO_ROOT,
        env={**os.environ, "SCAN_INVENTORY": str(_inventory_for(root)), "SCAN_REPORT": str(report)},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _plant(root: Path, relative: str, source: Path) -> None:
    target: Final = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)


def test_bundled_boringssl_grpcio_fails_the_fips_gate(tmp_path: Path) -> None:
    root: Final = tmp_path / "image"
    _plant(root, CYGRPC_INVENTORY_PATH, CYGRPC_SO)
    result: Final = _run_scan(root, tmp_path / "scan-report.txt")
    assert result.returncode == 1, result.stdout + result.stderr
    violations: Final = tuple(line for line in result.stdout.splitlines() if line.startswith("VIOLATION "))
    assert any("cygrpc" in line and "BoringSSL" in line for line in violations), result.stdout
    assert not any("cygrpc" in line for line in result.stdout.splitlines() if "WAIVED" in line), result.stdout


def test_system_openssl_linked_grpcio_at_its_inventory_path_is_clean(tmp_path: Path) -> None:
    ssl_extension: Final = Path(_ssl.__file__)
    headers: Final = subprocess.run(
        ["objdump", "-p", str(ssl_extension)], capture_output=True, text=True, timeout=60, check=True
    ).stdout
    assert "libcrypto.so.3" in headers, (
        f"_ssl at {ssl_extension} is not linked against system libcrypto.so.3, "
        f"so it cannot stand in for a system-OpenSSL cygrpc: {headers}"
    )
    root: Final = tmp_path / "image"
    _plant(root, CYGRPC_INVENTORY_PATH, ssl_extension)
    report: Final = tmp_path / "scan-report.txt"
    result: Final = _run_scan(root, report)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: all 1 ELF binaries accounted for" in result.stdout, result.stdout
    assert any(line.startswith("CLEAN ") and "cygrpc" in line for line in report.read_text().splitlines()), (
        report.read_text()
    )


def test_unreviewed_elf_outside_the_inventory_fails_the_gate(tmp_path: Path) -> None:
    root: Final = tmp_path / "image"
    _plant(root, "app/.venv/lib/python3.13/site-packages/somepkg/native.so", Path(_ssl.__file__))
    result: Final = _run_scan(root, tmp_path / "scan-report.txt")
    assert result.returncode == 1, result.stdout + result.stderr
    violations: Final = tuple(line for line in result.stdout.splitlines() if line.startswith("VIOLATION "))
    assert any("[unreviewed]" in line and "native.so" in line for line in violations), result.stdout


def test_empty_tree_passes(tmp_path: Path) -> None:
    root: Final = tmp_path / "image"
    root.mkdir()
    result: Final = _run_scan(root, tmp_path / "scan-report.txt")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: all 0 ELF binaries" in result.stdout, result.stdout
