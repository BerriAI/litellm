import _hashlib
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Final
from unittest.mock import Mock

import pytest
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import hashes

from litellm.proxy.common_utils.fips import (
    FipsModeError,
    cryptography_enforces_fips,
    enforce_fips_database,
    enforce_fips_mode,
    is_fips_mode,
    openssl_enforces_fips,
)
from litellm.repositories.fips_repository import FipsRepository


def _fake_openssl(monkeypatch: pytest.MonkeyPatch, *, fips_flag: int, md5_refused: bool) -> None:
    real_new: Final = _hashlib.new

    def new(name: str, data: bytes = b"", *, usedforsecurity: bool = True) -> _hashlib.HASH:
        if name == "md5" and md5_refused:
            raise _hashlib.UnsupportedDigestmodError("unsupported hash type")
        return real_new(name, data, usedforsecurity=usedforsecurity)

    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: fips_flag)
    monkeypatch.setattr(_hashlib, "new", new)


@pytest.mark.parametrize(
    "general_settings, expected",
    [
        ({"fips_mode": True}, True),
        ({"fips_mode": " TRUE "}, True),
        ({"fips_mode": False}, False),
        ({"fips_mode": "false"}, False),
        ({"fips_mode": None}, False),
        ({}, False),
    ],
)
def test_is_fips_mode(general_settings: Mapping[str, object], expected: bool) -> None:
    assert is_fips_mode(general_settings) is expected


@pytest.mark.parametrize(
    "fips_flag, md5_refused, enforces",
    [(1, True, True), (0, True, False), (1, False, False)],
    ids=["fips", "fips-flag-off", "md5-allowed"],
)
def test_openssl_enforces_fips_needs_the_fips_flag_and_a_refused_md5(
    monkeypatch: pytest.MonkeyPatch, fips_flag: int, md5_refused: bool, enforces: bool
) -> None:
    _fake_openssl(monkeypatch, fips_flag=fips_flag, md5_refused=md5_refused)
    assert openssl_enforces_fips() is enforces


def test_openssl_without_get_fips_mode_does_not_enforce_fips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(_hashlib, "get_fips_mode")
    assert openssl_enforces_fips() is False


@pytest.mark.parametrize(
    "fips_enabled, sha256_broken, enforces",
    [(True, False, True), (False, False, False), (True, True, False)],
    ids=["fips", "fips-flag-off", "sha256-broken"],
)
def test_cryptography_enforces_fips_needs_the_fips_flag_and_a_working_sha256(
    monkeypatch: pytest.MonkeyPatch, fips_enabled: bool, sha256_broken: bool, enforces: bool
) -> None:
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: fips_enabled)
    if sha256_broken:
        monkeypatch.setattr(hashes, "Hash", Mock(side_effect=InternalError("Unknown OpenSSL error", [])))
    assert cryptography_enforces_fips() is enforces


def test_cryptography_without_is_fips_enabled_does_not_enforce_fips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(rust_openssl, "is_fips_enabled")
    assert cryptography_enforces_fips() is False


def test_this_runners_python_openssl_does_not_enforce_fips() -> None:
    if _hashlib.get_fips_mode() == 1:
        pytest.skip("this runner's Python OpenSSL is in FIPS mode")
    assert openssl_enforces_fips() is False


def test_this_runners_cryptography_openssl_does_not_enforce_fips() -> None:
    if rust_openssl.is_fips_enabled():
        pytest.skip("this runner's cryptography OpenSSL is in FIPS mode")
    assert cryptography_enforces_fips() is False


@pytest.mark.parametrize(
    "general_settings, openssl_fips, cryptography_fips, refusal",
    [
        ({}, False, False, None),
        ({"fips_mode": False}, False, False, None),
        ({"fips_mode": True}, True, True, None),
        ({"fips_mode": "true"}, True, True, None),
        ({"fips_mode": True}, False, True, "general_settings.fips_mode is on but OpenSSL does not enforce FIPS"),
        ({"fips_mode": True}, True, False, "cryptography library's OpenSSL has no working FIPS provider"),
        ({"fips_mode": "enforced"}, True, True, "general_settings.fips_mode='enforced' is not a boolean"),
        ({"fips_mode": 1}, True, True, "general_settings.fips_mode=1 is not a boolean"),
    ],
    ids=[
        "unset",
        "false",
        "fips",
        "fips-from-a-string",
        "openssl-not-fips",
        "cryptography-not-fips",
        "not-a-boolean",
        "a-number",
    ],
)
def test_enforce_fips_mode(
    monkeypatch: pytest.MonkeyPatch,
    general_settings: Mapping[str, object],
    openssl_fips: bool,
    cryptography_fips: bool,
    refusal: str | None,
) -> None:
    _fake_openssl(monkeypatch, fips_flag=int(openssl_fips), md5_refused=openssl_fips)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: cryptography_fips)

    with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
        enforce_fips_mode(general_settings)


class _Database:
    """Prisma's raw-query edge over a database whose only LiteLLM table besides LiteLLM_Config is the key table."""

    def __init__(self, *, marked_fips: bool, holds_rows: bool, readable: bool = True) -> None:
        self.marked_fips = marked_fips
        self.holds_rows = holds_rows
        self.readable = readable

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if not self.readable:
            raise RuntimeError('relation "LiteLLM_Config" does not exist')
        if "information_schema.tables" in query:
            return [{"table_name": "LiteLLM_VerificationToken"}]
        if '"LiteLLM_VerificationToken"' in query:
            return [{"marked_fips": self.marked_fips, "holds_rows": self.holds_rows}]
        return [{"?column?": 1}] if self.marked_fips else []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.marked_fips = True
        return 1


class _Prisma:
    def __init__(self, database: _Database) -> None:
        self.writer_db: Final = database


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fips_mode, database, refusal, marked_after",
    [
        (True, _Database(marked_fips=False, holds_rows=False), None, True),
        (True, _Database(marked_fips=True, holds_rows=True), None, True),
        (True, _Database(marked_fips=False, holds_rows=True), "already holds data written without it", False),
        (True, _Database(marked_fips=False, holds_rows=False, readable=False), "could not be checked", False),
        (False, _Database(marked_fips=True, holds_rows=True), "set up with general_settings.fips_mode: true", True),
        (False, _Database(marked_fips=False, holds_rows=True), None, False),
        (False, _Database(marked_fips=True, holds_rows=True, readable=False), None, True),
    ],
    ids=[
        "fips-on-new-database",
        "fips-on-fips-database",
        "fips-on-used-database",
        "fips-on-unreadable-database",
        "fips-off-fips-database",
        "fips-off-used-database",
        "fips-off-unreadable-database",
    ],
)
async def test_enforce_fips_database(
    fips_mode: bool, database: _Database, refusal: str | None, marked_after: bool
) -> None:
    with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
        await enforce_fips_database(FipsRepository(_Prisma(database)), fips_mode=fips_mode, database_is_configured=True)

    assert database.marked_fips is marked_after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fips_mode, database_is_configured, refusal",
    [(True, True, "can't be reached"), (True, False, None), (False, True, None)],
    ids=["fips-on-unreachable", "fips-on-no-database", "fips-off-unreachable"],
)
async def test_enforce_fips_database_without_a_connected_database(
    fips_mode: bool, database_is_configured: bool, refusal: str | None
) -> None:
    with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
        await enforce_fips_database(None, fips_mode=fips_mode, database_is_configured=database_is_configured)
