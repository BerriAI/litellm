import _hashlib
import logging
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from traceback import walk_tb
from typing import Final
from unittest.mock import Mock

import pytest
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import hashes

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.fips import (
    FipsModeError,
    cryptography_enforces_fips,
    database_password_login_warning,
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
    monkeypatch.setenv("LITELLM_SALT_KEY", _HEX_SALT_KEY)

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
    monkeypatch.setenv("LITELLM_SALT_KEY", _HEX_SALT_KEY)
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


@pytest.fixture
def fips_openssl_on_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_openssl(monkeypatch, fips_flag=1, scrypt_refused=True)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: True)


_SHORT_SALT_KEYS: Final = ("thirteen-byte", "é" * 6 + "x", "sk-0a1b2c3d4e", "")
_SHORT_SALT_KEY_IDS: Final = ("13-ascii-bytes", "13-utf8-bytes-in-7-characters", "sk-and-10-hex-digits", "empty")


@pytest.mark.parametrize(
    "salt_key, key",
    [
        (_HEX_SALT_KEY, _HEX_SALT_KEY.encode()),
        ("fourteen-bytes", b"fourteen-bytes"),
        ("é" * 7, "é".encode() * 7),
        ("\udcff" * 14, b"\xff" * 14),
    ],
    ids=[
        "hex-is-not-decoded",
        "14-ascii-bytes",
        "14-utf8-bytes-in-7-characters",
        "environment-bytes-that-are-not-utf8",
    ],
)
def test_key_derivation_key_is_the_utf8_bytes_of_the_salt_key(salt_key: str, key: bytes) -> None:
    assert key_derivation_key(salt_key) == key


@pytest.mark.parametrize(
    "salt_key", (*_SHORT_SALT_KEYS, "\udcff" * 13), ids=(*_SHORT_SALT_KEY_IDS, "13-environment-bytes")
)
def test_key_derivation_key_refuses_a_salt_key_under_112_bits(salt_key: str) -> None:
    assert key_derivation_key(salt_key) is None


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize(
    "salt_key",
    [_HEX_SALT_KEY, "fourteen-bytes", "é" * 7],
    ids=["openssl-rand-hex-32", "14-ascii-bytes", "14-utf8-bytes-in-7-characters"],
)
def test_fips_mode_starts_with_a_salt_key_of_at_least_112_bits(monkeypatch: pytest.MonkeyPatch, salt_key: str) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    enforce_fips_mode({"fips_mode": True})
    assert fips_mode_active() is True


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("salt_key", _SHORT_SALT_KEYS, ids=_SHORT_SALT_KEY_IDS)
def test_fips_mode_refuses_to_start_with_a_salt_key_under_112_bits(
    monkeypatch: pytest.MonkeyPatch, salt_key: str
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True})
    assert str(refused.value) == (
        "general_settings.fips_mode is on but LITELLM_SALT_KEY is too short. fips_mode derives its encryption keys "
        "from it, so it must be at least 112 bits (14 bytes). Set LITELLM_SALT_KEY=sk-$(openssl rand -hex 32), or "
        "turn general_settings.fips_mode off."
    )
    assert fips_mode_active() is False


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
def test_fips_mode_refuses_to_start_without_a_salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True})
    assert str(refused.value) == (
        "general_settings.fips_mode is on but LITELLM_SALT_KEY is not set. fips_mode derives its encryption keys "
        "from it and never from the master key, which also signs tokens. Set "
        "LITELLM_SALT_KEY=sk-$(openssl rand -hex 32), or turn general_settings.fips_mode off."
    )
    assert fips_mode_active() is False


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("salt_key", _SHORT_SALT_KEYS[:3], ids=_SHORT_SALT_KEY_IDS[:3])
def test_the_fips_mode_salt_key_refusal_does_not_print_the_key(monkeypatch: pytest.MonkeyPatch, salt_key: str) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True})
    assert salt_key not in str(refused.value)
    assert salt_key[3:] not in str(refused.value)


_FRAME_SECRET: Final = "frame-secret"


@pytest.mark.parametrize(
    "openssl_fips, cryptography_fips",
    [(True, True), (False, True), (True, False)],
    ids=["salt-key-under-112-bits", "openssl-not-fips", "cryptography-not-fips"],
)
def test_a_refused_fips_mode_boot_leaves_the_salt_key_in_no_frame_local(
    monkeypatch: pytest.MonkeyPatch, openssl_fips: bool, cryptography_fips: bool
) -> None:
    _fake_openssl(monkeypatch, fips_flag=int(openssl_fips), scrypt_refused=openssl_fips)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: cryptography_fips)
    monkeypatch.setenv("LITELLM_SALT_KEY", _FRAME_SECRET)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_mode({"fips_mode": True})
    held_by: Final = tuple(
        frame.f_code.co_name for frame, _ in walk_tb(refused.tb) if _FRAME_SECRET in repr(frame.f_locals)
    )
    assert held_by == ()


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("general_settings", [{}, {"fips_mode": False}], ids=["unset", "off"])
@pytest.mark.parametrize(
    "salt_key",
    [_HEX_SALT_KEY, "thirteen-byte", "", None],
    ids=["openssl-rand-hex-32", "13-bytes", "empty", "no-key"],
)
def test_standard_mode_starts_whatever_the_salt_key_looks_like(
    monkeypatch: pytest.MonkeyPatch, general_settings: Mapping[str, object], salt_key: str | None
) -> None:
    if salt_key is None:
        monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    else:
        monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    enforce_fips_mode(general_settings)
    assert fips_mode_active() is False


_PASSWORD_DATABASE_URL: Final = "postgresql://litellm:hunter2-pw@db:5432/litellm?sslmode=verify-full"
_CERT_DATABASE_URL: Final = "postgresql://litellm@db:5432/litellm?sslidentity=/certs/client.p12&sslpassword=p12-pass"


@pytest.mark.parametrize(
    "environ",
    [
        {"DATABASE_URL": _CERT_DATABASE_URL},
        {},
        {"DATABASE_URL": _PASSWORD_DATABASE_URL, "IAM_TOKEN_DB_AUTH": "True"},
        {"DATABASE_URL": _PASSWORD_DATABASE_URL, "IAM_TOKEN_DB_AUTH": "1"},
        {"DATABASE_URL": _PASSWORD_DATABASE_URL, "AZURE_POSTGRESQL_AUTH": "true"},
        {"DATABASE_URL": _PASSWORD_DATABASE_URL, "AZURE_POSTGRESQL_AUTH": "yes"},
        {"DATABASE_URL": "postgresql://litellm:pw@[::1/litellm"},
        {"DATABASE_URL": _CERT_DATABASE_URL, "DATABASE_URL_READ_REPLICA": _CERT_DATABASE_URL},
        {
            "DATABASE_URL": _CERT_DATABASE_URL,
            "DATABASE_URL_READ_REPLICA": _PASSWORD_DATABASE_URL,
            "IAM_TOKEN_DB_AUTH": "true",
        },
        {"DATABASE_URL": _CERT_DATABASE_URL, "DIRECT_URL": _CERT_DATABASE_URL},
        {"DATABASE_URL": _PASSWORD_DATABASE_URL, "DIRECT_URL": _CERT_DATABASE_URL, "IAM_TOKEN_DB_AUTH": "true"},
    ],
    ids=[
        "client-certificate",
        "no-database",
        "rds-iam-token",
        "rds-iam-token-set-to-1",
        "azure-entra-token",
        "azure-entra-token-set-to-yes",
        "unparseable-url",
        "client-certificate-replica",
        "rds-iam-token-replica",
        "client-certificate-direct-url",
        "rds-iam-token-client-certificate-direct-url",
    ],
)
def test_no_password_login_warning_without_a_database_password(environ: Mapping[str, str]) -> None:
    assert database_password_login_warning(environ) is None


_POOLED_DATABASE_URL: Final = "postgresql://litellm_pgbouncer:pool-pw@127.0.0.1:6432/litellm?pgbouncer=true"


@pytest.mark.parametrize(
    "token_login",
    [{}, {"IAM_TOKEN_DB_AUTH": "true"}, {"AZURE_POSTGRESQL_AUTH": "true"}],
    ids=["password-upstream", "rds-iam-upstream", "azure-entra-upstream"],
)
def test_pgbouncer_workers_always_get_the_password_login_warning(token_login: Mapping[str, str]) -> None:
    warning: Final = database_password_login_warning(
        {"DATABASE_URL": _POOLED_DATABASE_URL, "LITELLM_PGBOUNCER_POOLED_DATABASE_URL": "true", **token_login}
    )
    assert warning is not None
    assert "LITELLM_PGBOUNCER_ENABLED" in warning
    assert "pool-pw" not in warning


def test_an_unreadable_token_flag_still_gets_the_password_login_warning() -> None:
    assert database_password_login_warning({"DATABASE_URL": _PASSWORD_DATABASE_URL, "IAM_TOKEN_DB_AUTH": "maybe"})


@pytest.mark.parametrize(
    "environ, named",
    [
        ({"DATABASE_URL": _PASSWORD_DATABASE_URL}, "DATABASE_URL logs in"),
        (
            {"DATABASE_URL": _CERT_DATABASE_URL, "DATABASE_URL_READ_REPLICA": _PASSWORD_DATABASE_URL},
            "DATABASE_URL_READ_REPLICA logs in",
        ),
        ({"DATABASE_URL_READ_REPLICA": _PASSWORD_DATABASE_URL}, "DATABASE_URL_READ_REPLICA logs in"),
        ({"DATABASE_URL": _CERT_DATABASE_URL, "DIRECT_URL": _PASSWORD_DATABASE_URL}, "DIRECT_URL logs in"),
        (
            {"DATABASE_URL": _PASSWORD_DATABASE_URL, "DIRECT_URL": _PASSWORD_DATABASE_URL, "IAM_TOKEN_DB_AUTH": "true"},
            "DIRECT_URL logs in",
        ),
        (
            {
                "DATABASE_URL": _PASSWORD_DATABASE_URL,
                "DIRECT_URL": _PASSWORD_DATABASE_URL,
                "AZURE_POSTGRESQL_AUTH": "true",
            },
            "DIRECT_URL logs in",
        ),
    ],
    ids=[
        "writer",
        "replica-with-certificate-writer",
        "replica-only",
        "direct-url-with-certificate-writer",
        "direct-url-with-rds-iam-token",
        "direct-url-with-azure-entra-token",
    ],
)
def test_the_password_login_warning_names_the_url_but_not_the_password(environ: Mapping[str, str], named: str) -> None:
    warning: Final = database_password_login_warning(environ)
    assert warning is not None
    assert named in warning
    assert "hunter2-pw" not in warning


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: tuple[str, ...] = ()

    def emit(self, record: logging.LogRecord) -> None:
        self.messages = (*self.messages, record.getMessage())


@pytest.fixture
def proxy_warnings() -> Iterator[_Records]:
    records: Final = _Records()
    level: Final = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.WARNING)
    verbose_proxy_logger.addHandler(records)
    yield records
    verbose_proxy_logger.removeHandler(records)
    verbose_proxy_logger.setLevel(level)


@pytest.mark.usefixtures("fips_openssl_on_both_sides")
@pytest.mark.parametrize("fips_mode, warned", [(True, True), (False, False)], ids=["fips-mode-on", "fips-mode-off"])
def test_only_fips_mode_warns_about_a_database_password_login(
    monkeypatch: pytest.MonkeyPatch, proxy_warnings: _Records, fips_mode: bool, warned: bool
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", _HEX_SALT_KEY)
    monkeypatch.setenv("DATABASE_URL", _PASSWORD_DATABASE_URL)
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    enforce_fips_mode({"fips_mode": fips_mode})
    assert any("logs in to Postgres with a password" in message for message in proxy_warnings.messages) is warned
