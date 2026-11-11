import _hashlib
import hmac
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from typing import Final
from unittest.mock import MagicMock, Mock

import jwt
import pytest
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import hashes

from litellm.proxy.auth.login_utils import encode_ui_session_jwt
from litellm.proxy.common_utils.fips import (
    FipsModeError,
    cryptography_enforces_fips,
    enforce_fips_mode,
    fips_mode_active,
    openssl_enforces_fips,
)
from litellm.types.proxy.ui_sso import ReturnedUITokenObject


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


_HMAC_KEY: Final = "k" * 64
_SHORT_HMAC_KEY: Final = "thirteen-byte"
_UI_TOKEN: Final = ReturnedUITokenObject(
    user_id="u",
    key="sk-1",
    user_email=None,
    user_role="proxy_admin",
    login_method="username_password",
    premium_user=False,
    auth_header_name="Authorization",
    disabled_non_admin_personal_key_creation=False,
    server_root_path="",
    password_reset_required=False,
)


@pytest.fixture
def hmac_digest(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    digest: Final = MagicMock(wraps=hmac.digest)
    monkeypatch.setattr("litellm.proxy.common_utils.fips_jwt.hmac", MagicMock(digest=digest))
    return digest


@pytest.mark.parametrize("algorithm, digest_name", [("HS256", "sha256"), ("HS384", "sha384"), ("HS512", "sha512")])
def test_fips_mode_computes_every_jwt_hmac_with_hmac_digest_and_the_same_bytes(
    fips_mode: None, hmac_digest: MagicMock, algorithm: str, digest_name: str
) -> None:
    token: Final = jwt.encode({"sub": "u"}, _HMAC_KEY, algorithm=algorithm)

    assert token == jwt.PyJWT().encode({"sub": "u"}, _HMAC_KEY, algorithm=algorithm)
    assert jwt.decode(token, _HMAC_KEY, algorithms=[algorithm]) == {"sub": "u"}
    assert [call.args[2] for call in hmac_digest.call_args_list] == [digest_name, digest_name]


def test_fips_mode_refuses_a_jwt_hmac_key_under_112_bits(fips_mode: None) -> None:
    token: Final = jwt.PyJWT().encode({"sub": "u"}, _SHORT_HMAC_KEY, algorithm="HS256")

    with pytest.raises(jwt.InvalidKeyError, match="at least 112 bits"):
        jwt.encode({"sub": "u"}, _SHORT_HMAC_KEY, algorithm="HS256")
    with pytest.raises(jwt.InvalidKeyError, match="at least 112 bits"):
        jwt.decode(token, _SHORT_HMAC_KEY, algorithms=["HS256"])


def test_turning_fips_mode_off_gives_jwt_its_own_hmac_back(fips_mode: None, hmac_digest: MagicMock) -> None:
    enforce_fips_mode({})

    token: Final = jwt.encode({"sub": "u"}, _SHORT_HMAC_KEY, algorithm="HS256")

    assert jwt.decode(token, _SHORT_HMAC_KEY, algorithms=["HS256"]) == {"sub": "u"}
    hmac_digest.assert_not_called()


def test_fips_mode_refuses_a_ui_session_cookie_signed_with_a_master_key_under_112_bits(fips_mode: None) -> None:
    with pytest.raises(jwt.InvalidKeyError, match="at least 112 bits"):
        encode_ui_session_jwt(_UI_TOKEN, _SHORT_HMAC_KEY)


def test_fips_mode_signs_the_ui_session_cookie_with_hmac_digest(fips_mode: None, hmac_digest: MagicMock) -> None:
    token: Final = encode_ui_session_jwt(_UI_TOKEN, _HMAC_KEY)

    assert jwt.PyJWT().decode(token, _HMAC_KEY, algorithms=["HS256"])["user_id"] == "u"
    assert [call.args[0] for call in hmac_digest.call_args_list] == [_HMAC_KEY.encode()]


def test_standard_mode_signs_the_ui_session_cookie_with_a_short_master_key() -> None:
    token: Final = encode_ui_session_jwt(_UI_TOKEN, _SHORT_HMAC_KEY)

    assert jwt.decode(token, _SHORT_HMAC_KEY, algorithms=["HS256"])["user_id"] == "u"
