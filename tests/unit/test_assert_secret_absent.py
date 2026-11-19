import base64
import importlib.util
import io
import sys
import urllib.parse
import zipfile
from collections.abc import Callable, Mapping
from itertools import chain, product
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "assert_secret_absent.py"
_spec: Final = importlib.util.spec_from_file_location("assert_secret_absent", _MODULE_PATH)
scanner: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = scanner
_spec.loader.exec_module(scanner)

SECRET: Final = "lic/ense+KEY:9f8e~~??>>~?7d6c5b4a3210-zz"
PREFIXES: Final = ("", "x", "xy")
ENCODERS: Final = (base64.b64encode, base64.urlsafe_b64encode)


def _zip(members: Mapping[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _blob(prefix: str) -> bytes:
    return f"{prefix}{SECRET}tail".encode()


def _planted_forms() -> tuple[str, ...]:
    encoded: Final = tuple(encode(_blob(prefix)).decode() for encode, prefix in product(ENCODERS, PREFIXES))
    return (SECRET, urllib.parse.quote(SECRET, safe=""), *encoded)


def _assert_nothing_leaked(output: str) -> None:
    windows: Final = frozenset(
        chain.from_iterable((form[i : i + 8] for i in range(len(form) - 7)) for form in _planted_forms())
    )
    assert not [window for window in windows if window in output]


def _run(root: Path, secret: str = SECRET) -> tuple[int, str]:
    out: Final = io.StringIO()
    code: Final = scanner.main((str(root),), {scanner.SECRET_VARIABLE: secret}, out)
    return code, out.getvalue()


def _encoded_blob(prefix: str, encode: Callable[[bytes], bytes]) -> bytes:
    return b"trace " + encode(_blob(prefix)) + b" end"


QUOTED: Final = urllib.parse.quote(SECRET, safe="")
PLACEMENTS: Final = MappingProxyType(
    {
        "raw in a log": {"proxy.log": f"boot with key {SECRET} ok".encode()},
        "url-encoded in a url": {"upstream.log": f"GET /verify_license/{QUOTED}".encode()},
        "base64 aligned": {"a.json": _encoded_blob("", base64.b64encode)},
        "base64 one byte in": {"a.json": _encoded_blob("x", base64.b64encode)},
        "base64 two bytes in": {"a.json": _encoded_blob("xy", base64.b64encode)},
        "base64url two bytes in": {"a.json": _encoded_blob("xy", base64.urlsafe_b64encode)},
        "inside a zip member": {"trace.zip": _zip({"network.json": f'{{"auth": "{SECRET}"}}'.encode()})},
        "inside a nested zip": {"outer.zip": _zip({"inner.zip": _zip({"env.txt": SECRET.encode()})})},
        "named after it in a nested zip": {
            "outer.zip": _zip({"inner.zip": _zip({f"dump-{QUOTED}.txt": SECRET.encode()})})
        },
        "in a file name": {f"run-{QUOTED}.log": b"nothing inside"},
    }
)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_every_planted_copy_blocks_the_upload_without_printing_the_secret(tmp_path: Path, placement: str) -> None:
    worker: Final = tmp_path / "gw0"
    worker.mkdir()
    for name, data in PLACEMENTS[placement].items():
        (worker / name).write_bytes(data)
    code, output = _run(tmp_path)
    assert code == 1
    assert "secret found:" in output
    _assert_nothing_leaked(output)


def test_clean_results_pass_and_count_every_file(tmp_path: Path) -> None:
    (tmp_path / "junit.xml").write_text("<testsuite/>")
    (tmp_path / "gw0").mkdir()
    (tmp_path / "gw0" / "proxy.log").write_text("LITELLM_LICENSE is not set")
    code, output = _run(tmp_path)
    assert code == 0
    assert "no copy of the secret in 2 files" in output


def test_an_unset_secret_has_nothing_to_find(tmp_path: Path) -> None:
    (tmp_path / "proxy.log").write_text(SECRET)
    assert _run(tmp_path, secret="")[0] == 0


def test_a_symlinked_directory_fails_closed(tmp_path: Path) -> None:
    outside: Final = tmp_path / "outside"
    outside.mkdir()
    results: Final = tmp_path / "results"
    results.mkdir()
    (results / "linked").symlink_to(outside, target_is_directory=True)
    code, output = _run(results)
    assert code == 1
    assert "directory symlink" in output


def test_an_unreadable_archive_fails_without_echoing_its_contents(tmp_path: Path) -> None:
    archive: Final = _zip({"env.txt": f"LITELLM_LICENSE={SECRET}".encode()}, zipfile.ZIP_STORED)
    at: Final = archive.index(SECRET.encode())
    (tmp_path / "trace.zip").write_bytes(archive[:at] + bytes([archive[at] ^ 0xFF]) + archive[at + 1 :])
    code, output = _run(tmp_path)
    assert code == 1
    assert "scan failed with BadZipFile" in output
    _assert_nothing_leaked(output)
