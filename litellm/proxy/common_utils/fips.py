import ctypes
import os
import re
from collections.abc import Mapping
from typing import Final

from litellm.proxy.auth.master_key_boot_check import SALT_KEY_ENV_VAR
from litellm.secret_managers.main import str_to_bool

FIPS_MODE_SETTING: Final = "fips_mode"
_SETTING: Final = f"general_settings.{FIPS_MODE_SETTING}"
_SALT_KEY_HEX: Final = re.compile(r"(?:sk-)?((?:[0-9a-fA-F]{2}){32,})")
_OSSL_PARAM_UTF8_STRING: Final = 4
_OSSL_PARAM_OCTET_STRING: Final = 5
_OSSL_PARAM_UNMODIFIED: Final = ctypes.c_size_t(-1).value
_DERIVED_KEY_BYTES: Final = 32


class _OsslParam(ctypes.Structure):
    _fields_ = (
        ("key", ctypes.c_char_p),
        ("data_type", ctypes.c_uint),
        ("data", ctypes.c_char_p),
        ("data_size", ctypes.c_size_t),
        ("return_size", ctypes.c_size_t),
    )


class FipsModeError(Exception):
    pass


def _fips_mode_setting(general_settings: Mapping[str, object]) -> bool | None:
    """general_settings.fips_mode as a boolean: False when the key is absent, None when it is not a boolean, including
    a key with no value."""
    if FIPS_MODE_SETTING not in general_settings:
        return False
    value: Final = general_settings[FIPS_MODE_SETTING]
    if isinstance(value, bool):
        return value
    return str_to_bool(value) if isinstance(value, str) else None


def openssl_enforces_fips() -> bool:
    """Python's OpenSSL runs with FIPS-only defaults and itself refuses MD5."""
    try:
        import _hashlib

        if _hashlib.get_fips_mode() != 1:
            return False
    except (ImportError, AttributeError, ValueError):
        return False
    try:
        _hashlib.new("md5", b"", usedforsecurity=True)
    except ValueError:
        return True
    return False


def cryptography_enforces_fips() -> bool:
    """The cryptography library's own OpenSSL runs with FIPS-only defaults and its FIPS provider can hash SHA-256."""
    try:
        from cryptography.exceptions import InternalError, UnsupportedAlgorithm
        from cryptography.hazmat.bindings._rust import openssl as rust_openssl
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        return False
    try:
        if not rust_openssl.is_fips_enabled():
            return False
        hashes.Hash(hashes.SHA256()).finalize()
    except (AttributeError, InternalError, UnsupportedAlgorithm):
        return False
    return True


def fips_mode_refusal(general_settings: Mapping[str, object]) -> str | None:
    """Why the proxy must not start: general_settings.fips_mode is not a boolean, or is on but Python's or the
    cryptography library's OpenSSL does not enforce FIPS. None when it may start."""
    setting: Final = _fips_mode_setting(general_settings)
    if setting is None:
        value: Final = general_settings.get(FIPS_MODE_SETTING)
        if value is None:
            return (
                f"{_SETTING} has no value: it is empty, or an os.environ/ reference to a variable that is not set. "
                "Set it to true or false, or remove it."
            )
        return f"{_SETTING}={value!r} is not a boolean. Set it to true or false, or remove it."
    if not setting:
        return None
    if not openssl_enforces_fips():
        return (
            f"{_SETTING} is on but OpenSSL does not enforce FIPS. Run the proxy on an OpenSSL with its FIPS "
            f"provider active and default_properties = fips=yes, or turn {_SETTING} off."
        )
    if not cryptography_enforces_fips():
        return (
            f"{_SETTING} is on but the cryptography library's OpenSSL has no working FIPS provider. Build it "
            "against the system OpenSSL (pip install --no-binary cryptography cryptography), or turn "
            f"{_SETTING} off."
        )
    return None


def enforce_fips_mode(general_settings: Mapping[str, object]) -> None:
    """Raise FipsModeError with the fips_mode_refusal reason, if there is one."""
    refusal: Final = fips_mode_refusal(general_settings)
    if refusal is not None:
        raise FipsModeError(refusal)


def _param(name: bytes, data_type: int, data: bytes) -> _OsslParam:
    return _OsslParam(name, data_type, data, len(data), _OSSL_PARAM_UNMODIFIED)


def _kbkdf_hmac_sha256(key: bytes, label: bytes, context: bytes) -> bytes:
    import _hashlib

    libcrypto: Final = ctypes.CDLL(_hashlib.__file__)
    kdf_fetch: Final = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)(
        ("EVP_KDF_fetch", libcrypto)
    )
    ctx_new: Final = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(("EVP_KDF_CTX_new", libcrypto))
    derive: Final = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_OsslParam)
    )(("EVP_KDF_derive", libcrypto))
    ctx_free: Final = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(("EVP_KDF_CTX_free", libcrypto))
    kdf_free: Final = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(("EVP_KDF_free", libcrypto))
    params: Final = (_OsslParam * 6)(
        _param(b"mac", _OSSL_PARAM_UTF8_STRING, b"HMAC"),
        _param(b"digest", _OSSL_PARAM_UTF8_STRING, b"SHA256"),
        _param(b"key", _OSSL_PARAM_OCTET_STRING, key),
        _param(b"salt", _OSSL_PARAM_OCTET_STRING, label),
        _param(b"info", _OSSL_PARAM_OCTET_STRING, context),
    )
    derived: Final = ctypes.create_string_buffer(_DERIVED_KEY_BYTES)
    kdf: Final = kdf_fetch(None, b"KBKDF", None)
    ctx: Final = ctx_new(kdf)
    try:
        if not ctx or derive(ctx, derived, _DERIVED_KEY_BYTES, params) != 1:
            raise ValueError("OpenSSL could not run KBKDF")
        return derived.raw
    finally:
        ctx_free(ctx)
        kdf_free(kdf)


def derive_key_from_salt_key(label: bytes, context: bytes) -> str:
    """Hex of 32 bytes derived from LITELLM_SALT_KEY under label and context by SP 800-108r1 KBKDF in counter mode
    with HMAC-SHA256, run by EVP_KDF in the OpenSSL that hashlib uses. Raises ValueError when LITELLM_SALT_KEY is not
    at least 32 bytes of hex, optionally prefixed with sk-."""
    salt_key: Final = _SALT_KEY_HEX.fullmatch(os.getenv(SALT_KEY_ENV_VAR, ""))
    if salt_key is None:
        raise ValueError(
            f"With OpenSSL enforcing FIPS, {SALT_KEY_ENV_VAR} must be at least 32 bytes of hex, such as "
            "sk-$(openssl rand -hex 32)"
        )
    return _kbkdf_hmac_sha256(bytes.fromhex(salt_key.group(1)), label, context).hex()
