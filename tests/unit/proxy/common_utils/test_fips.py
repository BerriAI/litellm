import _hashlib
from contextlib import nullcontext

import pytest
from cryptography.hazmat.bindings._rust import openssl as rust_openssl

import litellm
from litellm.proxy.common_utils.fips import (
    FipsModeError,
    cryptography_enforces_fips,
    enforce_fips_mode,
    is_fips_mode,
    openssl_enforces_fips,
)


def _fake_openssl(monkeypatch, *, fips_flag: int, md5_refused: bool) -> None:
    real_new = _hashlib.new

    def new(name, data=b"", **kwargs):
        if name == "md5" and md5_refused:
            raise _hashlib.UnsupportedDigestmodError("unsupported hash type")
        return real_new(name, data, **kwargs)

    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: fips_flag)
    monkeypatch.setattr(_hashlib, "new", new)


@pytest.mark.parametrize("value, expected", [("true", True), (" TRUE ", True), ("false", False), (None, False)])
def test_is_fips_mode(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("LITELLM_FIPS_MODE", raising=False)
    else:
        monkeypatch.setenv("LITELLM_FIPS_MODE", value)
    assert is_fips_mode() is expected


@pytest.mark.parametrize(
    "fips_flag, md5_refused, enforces",
    [(1, True, True), (0, True, False), (1, False, False)],
    ids=["fips", "fips-flag-off", "md5-allowed"],
)
def test_openssl_enforces_fips_needs_the_fips_flag_and_a_refused_md5(monkeypatch, fips_flag, md5_refused, enforces):
    _fake_openssl(monkeypatch, fips_flag=fips_flag, md5_refused=md5_refused)
    assert openssl_enforces_fips() is enforces


def test_openssl_without_get_fips_mode_does_not_enforce_fips(monkeypatch):
    monkeypatch.delattr(_hashlib, "get_fips_mode")
    assert openssl_enforces_fips() is False


@pytest.mark.parametrize("fips_enabled", [True, False])
def test_cryptography_enforces_fips_reads_its_openssl_fips_flag(monkeypatch, fips_enabled):
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: fips_enabled)
    assert cryptography_enforces_fips() is fips_enabled


def test_cryptography_without_is_fips_enabled_does_not_enforce_fips(monkeypatch):
    monkeypatch.delattr(rust_openssl, "is_fips_enabled")
    assert cryptography_enforces_fips() is False


def test_this_runners_openssl_does_not_enforce_fips():
    if _hashlib.get_fips_mode() == 1:
        pytest.skip("this runner's OpenSSL is in FIPS mode")
    assert openssl_enforces_fips() is False
    assert cryptography_enforces_fips() is False


@pytest.mark.parametrize(
    "fips_mode, ssl_verify_env, ssl_verify_setting, openssl_fips, cryptography_fips, refusal",
    [
        (None, "false", False, False, False, None),
        ("", "false", False, False, False, None),
        ("false", "false", False, False, False, None),
        ("true", None, True, True, True, None),
        ("true", "false", True, True, True, "TLS certificate verification is disabled"),
        ("true", None, False, True, True, "TLS certificate verification is disabled"),
        ("true", None, True, False, True, "is on but OpenSSL does not enforce FIPS"),
        ("true", None, True, True, False, "cryptography library's OpenSSL does not enforce FIPS"),
        ("enforced", None, True, True, True, "LITELLM_FIPS_MODE='enforced' is not a boolean"),
    ],
    ids=[
        "unset",
        "empty",
        "false",
        "fips",
        "ssl-verify-env-off",
        "ssl-verify-setting-off",
        "openssl-not-fips",
        "cryptography-not-fips",
        "not-a-boolean",
    ],
)
def test_enforce_fips_mode(
    monkeypatch, fips_mode, ssl_verify_env, ssl_verify_setting, openssl_fips, cryptography_fips, refusal
):
    for name, value in (("LITELLM_FIPS_MODE", fips_mode), ("SSL_VERIFY", ssl_verify_env)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(litellm, "ssl_verify", ssl_verify_setting)
    _fake_openssl(monkeypatch, fips_flag=int(openssl_fips), md5_refused=openssl_fips)
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: cryptography_fips)

    with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
        enforce_fips_mode()
