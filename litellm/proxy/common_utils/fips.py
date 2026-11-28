import os
import threading
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import exclude_rsa_key_exchange_and_chacha20
from litellm.proxy.auth.master_key_boot_check import SALT_KEY_ENV_VAR
from litellm.proxy.db.pgbouncer import database_url_is_pooled
from litellm.proxy.db.token_auth import (
    AZURE_POSTGRESQL_AUTH_ENV_VAR,
    IAM_TOKEN_DB_AUTH_ENV_VAR,
    token_auth_flag_enabled,
)
from litellm.secret_managers.main import str_to_bool

FIPS_MODE_SETTING: Final = "fips_mode"
_KEY_DERIVATION_KEY_MIN_BYTES: Final = 14
KEY_DERIVATION_KEY_RULE: Final = (
    f"at least {_KEY_DERIVATION_KEY_MIN_BYTES * 8} bits ({_KEY_DERIVATION_KEY_MIN_BYTES} bytes)"
)
_SETTING: Final = f"general_settings.{FIPS_MODE_SETTING}"
_NEW_SALT_KEY: Final = f"{SALT_KEY_ENV_VAR}=sk-$(openssl rand -hex 32)"
_FIPS_MODE_ACTIVE: Final = threading.Event()


class FipsModeError(Exception):
    pass


def key_derivation_key(salt_key: str) -> bytes | None:
    """The key fips_mode derives its keys from, the UTF-8 bytes of salt_key as given (bytes an environment variable
    could not decode are kept as they were). None when that is under 112 bits (14 bytes)."""
    key: Final = salt_key.encode("utf-8", "surrogateescape")
    return key if len(key) >= _KEY_DERIVATION_KEY_MIN_BYTES else None


def fips_mode_setting(general_settings: Mapping[str, object]) -> bool | None:
    """general_settings.fips_mode as a boolean: False when the key is absent, None when it is not a boolean, including
    a key with no value."""
    if FIPS_MODE_SETTING not in general_settings:
        return False
    value: Final = general_settings[FIPS_MODE_SETTING]
    if isinstance(value, bool):
        return value
    return str_to_bool(value) if isinstance(value, str) else None


def openssl_enforces_fips() -> bool:
    """Python's OpenSSL runs with FIPS-only defaults and itself refuses scrypt."""
    try:
        import _hashlib

        if _hashlib.get_fips_mode() != 1:
            return False
    except (ImportError, AttributeError, ValueError):
        return False
    try:
        _hashlib.scrypt(b"", salt=b"\0" * 16, n=2, r=1, p=1, dklen=16)
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
    cryptography library's OpenSSL does not enforce FIPS, or LITELLM_SALT_KEY is not set or gives a key_derivation_key
    under 112 bits. None when it may start."""
    setting: Final = fips_mode_setting(general_settings)
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
    salt_key: Final = os.getenv(SALT_KEY_ENV_VAR)
    if salt_key is None:
        return (
            f"{_SETTING} is on but {SALT_KEY_ENV_VAR} is not set. fips_mode derives its encryption keys from it and "
            f"never from the master key, which also signs tokens. Set {_NEW_SALT_KEY}, or turn {_SETTING} off."
        )
    if key_derivation_key(salt_key) is None:
        return (
            f"{_SETTING} is on but {SALT_KEY_ENV_VAR} is too short. fips_mode derives its encryption keys from it, "
            f"so it must be {KEY_DERIVATION_KEY_RULE}. Set {_NEW_SALT_KEY}, or turn {_SETTING} off."
        )
    return None


def _token_login(environ: Mapping[str, str]) -> bool:
    """Whether RDS IAM or Azure Entra token login is on, read the way the token refresh reads it."""
    try:
        return any(
            token_auth_flag_enabled(environ.get(name), env_var=name)
            for name in (IAM_TOKEN_DB_AUTH_ENV_VAR, AZURE_POSTGRESQL_AUTH_ENV_VAR)
        )
    except ValueError:
        return False


def _password_login(environ: Mapping[str, str], name: str) -> bool:
    """Whether the Postgres URL in the env var carries a password."""
    try:
        return bool(urlsplit(environ.get(name, "")).password)
    except ValueError:
        return False


def database_password_login_warning(environ: Mapping[str, str]) -> str | None:
    """A warning when DATABASE_URL or DATABASE_URL_READ_REPLICA logs in with a password: to Postgres when neither
    RDS IAM nor Azure Entra token login is on, or to the in-container PgBouncer, which always takes a password. Also
    when DIRECT_URL, which the migrations log in with, carries a password, since token login never replaces it. None
    otherwise."""
    if database_url_is_pooled(environ) and _password_login(environ, "DATABASE_URL"):
        return (
            f"{_SETTING} is on but LITELLM_PGBOUNCER_ENABLED logs the workers in to the in-container PgBouncer with a "
            "password, and that login (SCRAM) is computed outside the FIPS module. Turn LITELLM_PGBOUNCER_ENABLED off "
            "and use client-certificate login: cert in pg_hba.conf, sslidentity and sslpassword in DATABASE_URL with "
            "no password, and PGSSLCERT and PGSSLKEY set to the same certificate and key."
        )
    token_login: Final = _token_login(environ)
    urls: Final = ("DIRECT_URL",) if token_login else ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL")
    url: Final = next((name for name in urls if _password_login(environ, name)), None)
    if url is None:
        return None
    return (
        f"{_SETTING} is on but {url} logs in to Postgres with a password. The password login (SCRAM) is "
        "computed outside the FIPS module. Use client-certificate login instead: cert in pg_hba.conf, sslidentity "
        f"and sslpassword in {url} with no password, and PGSSLCERT and PGSSLKEY set to the same certificate "
        "and key. "
        + (
            "Or leave DIRECT_URL unset, so the migrations log in the way DATABASE_URL does."
            if url == "DIRECT_URL"
            else "RDS IAM (IAM_TOKEN_DB_AUTH) and Azure Entra (AZURE_POSTGRESQL_AUTH) token login also avoid it."
        )
    )


def enforce_fips_mode(general_settings: Mapping[str, object]) -> None:
    refusal: Final = fips_mode_refusal(general_settings)
    if refusal is not None:
        raise FipsModeError(refusal)
    active: Final = fips_mode_setting(general_settings) is True
    exclude_rsa_key_exchange_and_chacha20(active)
    if active:
        _FIPS_MODE_ACTIVE.set()
        password_login: Final = database_password_login_warning(os.environ)
        if password_login is not None:
            verbose_proxy_logger.warning(password_login)
    else:
        _FIPS_MODE_ACTIVE.clear()


def fips_mode_active() -> bool:
    """Whether general_settings.fips_mode was on when enforce_fips_mode last accepted it."""
    return _FIPS_MODE_ACTIVE.is_set()
