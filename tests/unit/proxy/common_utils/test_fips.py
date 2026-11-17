import _hashlib
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import hashes

from litellm.proxy.common_utils.fips import (
    FipsModeError,
    cryptography_enforces_fips,
    enforce_fips_mode,
    fips_mode_active,
    openssl_enforces_fips,
)


@pytest.fixture(autouse=True)
def _fips_mode_off_after_each_test() -> Iterator[None]:
    yield
    enforce_fips_mode({})


def _fake_openssl(monkeypatch: pytest.MonkeyPatch, *, fips_flag: int, scrypt_refused: bool) -> None:
    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: fips_flag)
    if scrypt_refused:
        monkeypatch.setattr(_hashlib, "scrypt", Mock(side_effect=ValueError("[digital envelope routines] unsupported")))
    else:
        monkeypatch.setattr(_hashlib, "scrypt", Mock(return_value=b"\0" * 16))


@pytest.mark.parametrize(
    "fips_flag, scrypt_refused, enforces",
    [(1, True, True), (0, True, False), (1, False, False)],
    ids=["fips", "fips-flag-off", "scrypt-allowed"],
)
def test_openssl_enforces_fips_needs_the_fips_flag_and_a_refused_scrypt(
    monkeypatch: pytest.MonkeyPatch, fips_flag: int, scrypt_refused: bool, enforces: bool
) -> None:
    _fake_openssl(monkeypatch, fips_flag=fips_flag, scrypt_refused=scrypt_refused)
    assert openssl_enforces_fips() is enforces


def test_openssl_enforces_fips_while_md5_is_still_served_for_non_security_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    monkeypatch.setattr(_hashlib, "new", Mock(return_value=Mock()))
    assert openssl_enforces_fips() is True


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
        ({"fips_mode": "false"}, False, False, None),
        ({"fips_mode": True}, True, True, None),
        ({"fips_mode": "true"}, True, True, None),
        ({"fips_mode": True}, False, True, "general_settings.fips_mode is on but OpenSSL does not enforce FIPS"),
        ({"fips_mode": " TRUE "}, False, True, "general_settings.fips_mode is on but OpenSSL does not enforce FIPS"),
        ({"fips_mode": True}, True, False, "cryptography library's OpenSSL has no working FIPS provider"),
        ({"fips_mode": "enforced"}, True, True, "general_settings.fips_mode='enforced' is not a boolean"),
        ({"fips_mode": 1}, True, True, "general_settings.fips_mode=1 is not a boolean"),
        ({"fips_mode": None}, True, True, "general_settings.fips_mode has no value"),
    ],
    ids=[
        "unset",
        "false",
        "false-from-a-string",
        "fips",
        "fips-from-a-string",
        "openssl-not-fips",
        "openssl-not-fips-from-a-padded-string",
        "cryptography-not-fips",
        "not-a-boolean",
        "a-number",
        "no-value",
    ],
)
def test_enforce_fips_mode(
    monkeypatch: pytest.MonkeyPatch,
    general_settings: Mapping[str, object],
    openssl_fips: bool,
    cryptography_fips: bool,
    refusal: str | None,
) -> None:
    _fake_openssl(monkeypatch, fips_flag=int(openssl_fips), scrypt_refused=openssl_fips)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: cryptography_fips)

    with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
        enforce_fips_mode(general_settings)


@pytest.mark.parametrize(
    "general_settings, active",
    [({"fips_mode": True}, True), ({"fips_mode": "true"}, True), ({"fips_mode": False}, False), ({}, False)],
    ids=["on", "on-from-a-string", "off", "unset"],
)
def test_fips_mode_active_follows_the_setting_the_gate_accepted(
    monkeypatch: pytest.MonkeyPatch, general_settings: Mapping[str, object], active: bool
) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: True)
    enforce_fips_mode(general_settings)
    assert fips_mode_active() is active


def test_fips_mode_stays_inactive_on_a_refused_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=0, scrypt_refused=False)
    with pytest.raises(FipsModeError):
        enforce_fips_mode({"fips_mode": True})
    assert fips_mode_active() is False


def test_a_fips_openssl_alone_does_not_activate_fips_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    assert fips_mode_active() is False
