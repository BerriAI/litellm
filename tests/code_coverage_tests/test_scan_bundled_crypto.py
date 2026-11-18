import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SCAN_SCRIPT: Final = REPO_ROOT / "litellm/deploy/docker/scan_bundled_crypto.sh"
PASS_ONE: Final = "PASS: all 1 ELF binaries accounted for (inventoried, system-linked, or waived)\n"
FAIL_ONE: Final = "FAIL: 1 of 1 ELF binaries unaccounted for (see above)\n"


def _elf(path: Path, payload: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF" + b"\x00" * 12 + payload)
    return path


def _scan(
    tmp_path: Path, roots: Sequence[Path], inventory: Sequence[str] = (), emit: bool = False
) -> tuple[subprocess.CompletedProcess[str], Path]:
    inventory_file: Final = tmp_path / "inventory.txt"
    inventory_file.write_text("".join(f"{line}\n" for line in inventory))
    report: Final = tmp_path / "report.txt"
    env: Final = {
        **{k: v for k, v in os.environ.items() if not k.startswith("SCAN_")},
        "SCAN_INVENTORY": str(inventory_file),
        "SCAN_REPORT": str(report),
        **({"SCAN_EMIT_INVENTORY": "1"} if emit else {}),
    }
    result: Final = subprocess.run(
        ["sh", str(SCAN_SCRIPT), *map(str, roots)], env=env, capture_output=True, text=True, check=False
    )
    return result, report


def test_inventoried_crypto_free_binary_passes(tmp_path: Path) -> None:
    clean: Final = _elf(tmp_path / "root" / "pkg" / "clean.so")
    (tmp_path / "root" / "pkg" / "README.txt").write_text("not an ELF")

    result, report = _scan(tmp_path, [tmp_path / "root"], inventory=[str(clean)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == PASS_ONE
    assert report.read_text() == f"CLEAN {clean}\nscanned 1 ELF binaries\n"


def test_unreviewed_binary_fails(tmp_path: Path) -> None:
    unknown: Final = _elf(tmp_path / "root" / "pkg" / "unknown.so")

    result, _ = _scan(tmp_path, [tmp_path / "root"])

    assert result.returncode == 1
    assert result.stdout == (
        f"VIOLATION {unknown} [unreviewed] not in fips_elf_inventory.txt; review it, then regenerate the inventory\n"
        + FAIL_ONE
    )


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        (b"BoringSSL", "BoringSSL"),
        (b"AWS-LC", "AWS-LC"),
        (b"OpenSSL 3.0.13 30 Jan 2024", "bundled-OpenSSL"),
        (b"sodium_init", "libsodium"),
        (b"ring_core_0_17_8_", "ring"),
        (b"rustls", "rustls"),
        (b"mbedtls", "mbedTLS"),
        (b"wolfSSL", "wolfSSL"),
        (b"libgcrypt", "libgcrypt"),
    ],
)
def test_bundled_crypto_without_waiver_fails_even_when_inventoried(tmp_path: Path, payload: bytes, label: str) -> None:
    bundled: Final = _elf(tmp_path / "root" / "pkg" / "native.so", payload)

    result, report = _scan(tmp_path, [tmp_path / "root"], inventory=[str(bundled)])

    assert result.returncode == 1
    assert result.stdout == f"VIOLATION {bundled} [{label}] bundled crypto with no waiver\n" + FAIL_ONE
    assert report.read_text() == f"BUNDLED-CRYPTO {bundled} [{label}]\n"


@pytest.mark.parametrize(
    "relative",
    [
        "site-packages/grpc/_cython/cygrpc.cpython-313-x86_64-linux-gnu.so",
        "site-packages/granian/_granian.cpython-313-aarch64-linux-gnu.so",
        "site-packages/litellm/rust_bridge/_native.abi3.so",
        "site-packages/psycopg_binary.libs/libcrypto-2d1629ee.so.3",
    ],
)
def test_waived_bundled_crypto_passes(tmp_path: Path, relative: str) -> None:
    waived: Final = _elf(tmp_path / "root" / relative, b"BoringSSL")

    result, report = _scan(tmp_path, [tmp_path / "root"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"WAIVED    {waived} [BoringSSL]\n" + PASS_ONE
    assert report.read_text() == f"BUNDLED-CRYPTO {waived} [BoringSSL]\nscanned 1 ELF binaries\n"


@pytest.mark.parametrize(
    "relative",
    [
        "site-packages/granian_plugin/native.so",
        "site-packages/litellm/rust_bridge/other.abi3.so",
        "site-packages/libsodium/libsodium.so",
        "site-packages/xmlsec.libs/libcrypto.so.3",
    ],
)
def test_bundled_crypto_next_to_a_waiver_still_fails(tmp_path: Path, relative: str) -> None:
    bundled: Final = _elf(tmp_path / "root" / relative, b"BoringSSL")

    result, _ = _scan(tmp_path, [tmp_path / "root"])

    assert result.returncode == 1
    assert result.stdout == f"VIOLATION {bundled} [BoringSSL] bundled crypto with no waiver\n" + FAIL_ONE


def test_cryptography_wheel_with_its_own_openssl_fails(tmp_path: Path) -> None:
    wheel: Final = _elf(
        tmp_path / "root" / "site-packages/cryptography/hazmat/bindings/_rust.abi3.so", b"OpenSSL 4.0.1 9 Jun 2026"
    )

    result, _ = _scan(tmp_path, [tmp_path / "root"], inventory=[str(wheel)])

    assert result.returncode == 1
    assert result.stdout == f"VIOLATION {wheel} [bundled-OpenSSL] bundled crypto with no waiver\n" + FAIL_ONE


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"], ids=["intel", "arm"])
def test_inventory_entry_matches_any_arch_and_hash_suffix(tmp_path: Path, arch: str) -> None:
    root: Final = tmp_path / "root"
    _elf(root / "pkg" / f"_mod.cpython-313-{arch}-linux-gnu.so")
    _elf(root / "numpy.libs" / "libgfortran-040039e1-0352e75f.so.5.0.0")

    result, _ = _scan(
        tmp_path,
        [root],
        inventory=[
            f"{root}/pkg/_mod.cpython-313-ARCH-linux-gnu.so",
            f"{root}/numpy.libs/libgfortran-040039e1.so.5.0.0",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "PASS: all 2 ELF binaries accounted for (inventoried, system-linked, or waived)\n"


@pytest.mark.parametrize("emit", [False, True], ids=["scan", "emit"])
def test_missing_root_fails_before_scanning(tmp_path: Path, emit: bool) -> None:
    clean: Final = _elf(tmp_path / "root" / "clean.so")
    missing: Final = tmp_path / "missing"

    result, report = _scan(tmp_path, [tmp_path / "root", missing], inventory=[str(clean)], emit=emit)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"FAIL: scan root {missing} does not exist\n"
    assert not report.exists()


def test_no_roots_fails(tmp_path: Path) -> None:
    result, report = _scan(tmp_path, [])

    assert result.returncode == 1
    assert result.stderr == "FAIL: no scan roots given\n"
    assert not report.exists()


def test_root_with_no_elf_binaries_fails(tmp_path: Path) -> None:
    root: Final = tmp_path / "root"
    root.mkdir()
    (root / "notes.txt").write_text("not an ELF")

    result, _ = _scan(tmp_path, [root])

    assert result.returncode == 1
    assert result.stdout == f"FAIL: no ELF binaries found under {root}\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 directory, so find raises no error")
def test_unreadable_directory_fails_the_scan(tmp_path: Path) -> None:
    clean: Final = _elf(tmp_path / "root" / "clean.so")
    locked: Final = tmp_path / "root" / "locked"
    _elf(locked / "hidden.so", b"BoringSSL")
    locked.chmod(0)
    try:
        result, report = _scan(tmp_path, [tmp_path / "root"], inventory=[str(clean)])
    finally:
        locked.chmod(0o755)

    assert result.returncode != 0
    assert "PASS" not in result.stdout
    assert str(locked) in result.stderr
    assert not report.exists()


def test_emit_mode_prints_sorted_normalized_elf_paths(tmp_path: Path) -> None:
    root: Final = tmp_path / "root"
    _elf(root / "b" / "_mod.cpython-313-x86_64-linux-gnu.so")
    _elf(root / "a.libs" / "libfoo-0123abcd.so")
    (root / "a.libs" / "LICENSE").write_text("not an ELF")

    result, report = _scan(tmp_path, [root], emit=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{root}/a.libs/libfoo.so\n{root}/b/_mod.cpython-313-ARCH-linux-gnu.so\n"
    assert not report.exists()
