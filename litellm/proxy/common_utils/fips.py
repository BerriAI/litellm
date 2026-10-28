import os
from typing import Final

from litellm.secret_managers.main import str_to_bool

FIPS_MODE_ENV_VAR: Final = "LITELLM_FIPS_MODE"


class FipsModeError(Exception):
    pass


def is_fips_mode() -> bool:
    return str_to_bool(os.getenv(FIPS_MODE_ENV_VAR)) is True


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


def enforce_fips_mode() -> None:
    """Refuse to start when LITELLM_FIPS_MODE is not a boolean, or is on but TLS verification is off
    or Python's or the cryptography library's OpenSSL does not enforce FIPS."""
    value: Final = os.getenv(FIPS_MODE_ENV_VAR)
    if value and str_to_bool(value) is None:
        raise FipsModeError(f"{FIPS_MODE_ENV_VAR}={value!r} is not a boolean. Set it to true or false, or unset it.")
    if not is_fips_mode():
        return

    from litellm.llms.custom_httpx.http_handler import get_ssl_verify

    get_ssl_verify()  # raises FipsModeError when SSL_VERIFY or litellm_settings.ssl_verify turns verification off
    if not openssl_enforces_fips():
        raise FipsModeError(
            f"{FIPS_MODE_ENV_VAR} is on but OpenSSL does not enforce FIPS. Run the proxy on an OpenSSL with "
            f"its FIPS provider active and default_properties = fips=yes, or unset {FIPS_MODE_ENV_VAR}."
        )
    if not cryptography_enforces_fips():
        raise FipsModeError(
            f"{FIPS_MODE_ENV_VAR} is on but the cryptography library's OpenSSL has no working FIPS provider. Build it "
            "against the system OpenSSL (pip install --no-binary cryptography cryptography), or unset "
            f"{FIPS_MODE_ENV_VAR}."
        )
