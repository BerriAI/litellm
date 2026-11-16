import _hashlib
from collections.abc import Iterator, Mapping
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
    enforce_fips_mode,
    fips_mode_active,
    key_derivation_key,
    openssl_enforces_fips,
)

_HEX_SALT_KEY: Final = "sk-" + bytes(range(32)).hex()


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
        enforce_fips_mode(general_settings, _HEX_SALT_KEY)


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
    enforce_fips_mode(general_settings, _HEX_SALT_KEY)
    assert fips_mode_active() is active


def test_fips_mode_stays_inactive_on_a_refused_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=0, scrypt_refused=False)
    with pytest.raises(FipsModeError):
        enforce_fips_mode({"fips_mode": True})
    assert fips_mode_active() is False


def test_a_fips_openssl_alone_does_not_activate_fips_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    assert fips_mode_active() is False


@pytest.fixture
def fips_openssl_on_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: True)


_SHORT_SALT_KEYS: Final = ("thirteen-byte", "é" * 6 + "x", "sk-0a1b2c3d4e", "")
_SHORT_SALT_KEY_IDS: Final = ("13-ascii-bytes", "13-utf8-bytes-in-7-characters", "sk-and-10-hex-digits", "empty")


@pytest.mark.parametrize(
    "salt_key, key",
    [
        (_HEX_SALT_KEY, bytes(range(32))),
        (bytes(range(32)).hex(), bytes(range(32))),
        ("sk-" + bytes(range(32)).hex().upper(), bytes(range(32))),
        ("sk-" + bytes(range(48)).hex(), bytes(range(48))),
        ("fourteen-bytes", b"fourteen-bytes"),
        ("é" * 7, "é".encode() * 7),
        ("sk-" + bytes(range(31)).hex(), ("sk-" + bytes(range(31)).hex()).encode()),
        (_HEX_SALT_KEY + "a", (_HEX_SALT_KEY + "a").encode()),
        (_HEX_SALT_KEY + "\n", (_HEX_SALT_KEY + "\n").encode()),
        ("SK-" + bytes(range(32)).hex(), ("SK-" + bytes(range(32)).hex()).encode()),
        ("\udcff" * 14, b"\xff" * 14),
    ],
    ids=[
        "sk-prefixed-hex",
        "bare-hex",
        "uppercase-hex-digits",
        "48-bytes-of-hex",
        "14-ascii-bytes",
        "14-utf8-bytes-in-7-characters",
        "sk-and-31-bytes-of-hex",
        "odd-hex-digit-count",
        "hex-and-a-trailing-newline",
        "uppercase-prefix",
        "environment-bytes-that-are-not-utf8",
    ],
)
def test_key_derivation_key_decodes_a_hex_salt_key_and_takes_any_other_as_its_bytes(salt_key: str, key: bytes) -> None:
    assert key_derivation_key(salt_key) == key


@pytest.mark.parametrize(
    "salt_key", (*_SHORT_SALT_KEYS, "\udcff" * 13), ids=(*_SHORT_SALT_KEY_IDS, "13-environment-bytes")
)
def test_key_derivation_key_refuses_a_salt_key_under_112_bits(salt_key: str) -> None:
    assert key_derivation_key(salt_key) is None


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize(
    "salt_key",
    [
        _HEX_SALT_KEY,
        bytes(range(32)).hex(),
        "fourteen-bytes",
        "é" * 7,
        "sk-" + bytes(range(31)).hex(),
        _HEX_SALT_KEY + "\n",
        "SK-" + bytes(range(32)).hex(),
    ],
    ids=[
        "sk-prefixed-hex",
        "bare-hex",
        "14-ascii-bytes",
        "14-utf8-bytes-in-7-characters",
        "sk-and-31-bytes-of-hex",
        "hex-and-a-trailing-newline",
        "uppercase-prefix",
    ],
)
def test_fips_mode_starts_with_a_salt_key_of_at_least_112_bits(salt_key: str) -> None:
    enforce_fips_mode({"fips_mode": True}, salt_key)
    assert fips_mode_active() is True


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize(
    "salt_key_in_environment, checked",
    [(True, "LITELLM_SALT_KEY"), (False, "the master key (LITELLM_SALT_KEY is not set)")],
    ids=["salt-key", "master-key"],
)
@pytest.mark.parametrize("salt_key", _SHORT_SALT_KEYS, ids=_SHORT_SALT_KEY_IDS)
def test_fips_mode_refuses_to_start_with_a_salt_key_under_112_bits(
    monkeypatch: pytest.MonkeyPatch, salt_key: str, salt_key_in_environment: bool, checked: str
) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    if salt_key_in_environment:
        monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True}, salt_key)
    assert str(refused.value) == (
        f"general_settings.fips_mode is on but {checked} is too short. fips_mode derives its encryption keys from it, "
        "so it must be at least 112 bits (14 bytes). Set LITELLM_SALT_KEY=sk-$(openssl rand -hex 32), or turn "
        "general_settings.fips_mode off."
    )
    assert fips_mode_active() is False


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
def test_fips_mode_refuses_to_start_with_no_salt_key_or_master_key() -> None:
    with pytest.raises(FipsModeError, match="neither LITELLM_SALT_KEY nor a master key is set") as refused:
        enforce_fips_mode({"fips_mode": True}, None)
    assert "LITELLM_SALT_KEY=sk-$(openssl rand -hex 32)" in str(refused.value)
    assert fips_mode_active() is False


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("salt_key", _SHORT_SALT_KEYS[:3], ids=_SHORT_SALT_KEY_IDS[:3])
def test_the_fips_mode_salt_key_refusal_does_not_print_the_key(monkeypatch: pytest.MonkeyPatch, salt_key: str) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True}, salt_key)
    assert salt_key not in str(refused.value)
    assert salt_key[3:] not in str(refused.value)


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("general_settings", [{}, {"fips_mode": False}], ids=["unset", "off"])
@pytest.mark.parametrize(
    "salt_key",
    ["a-salt-key-that-is-not-hex", "thirteen-byte", "", None],
    ids=["not-hex", "13-bytes", "empty", "no-key"],
)
def test_standard_mode_starts_whatever_the_salt_key_looks_like(
    general_settings: Mapping[str, object], salt_key: str | None
) -> None:
    enforce_fips_mode(general_settings, salt_key)
    assert fips_mode_active() is False
