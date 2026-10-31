from collections.abc import Mapping
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.repositories.fips_repository import FIPS_MARKER, FipsRepository
from litellm.secret_managers.main import str_to_bool

FIPS_MODE_SETTING: Final = "fips_mode"
_SETTING: Final = f"general_settings.{FIPS_MODE_SETTING}"


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


def is_fips_mode(general_settings: Mapping[str, object]) -> bool:
    return _fips_mode_setting(general_settings) is True


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


async def _is_marked_fips(database: FipsRepository) -> bool:
    try:
        return await database.is_marked_fips()
    except Exception as error:  # noqa: BLE001  # without FIPS mode, an unreadable database starts as it did before
        verbose_proxy_logger.warning("Could not check whether the database was set up in FIPS mode: %s", error)
        return False


async def fips_database_refusal(
    database: FipsRepository | None, *, fips_mode: bool, database_is_configured: bool
) -> str | None:
    """Why the proxy must not start on its database: FIPS mode is off on a database set up in FIPS mode, or on over
    a database that can't be checked or already holds rows written without it. None when it may start."""
    if not fips_mode:
        if database is not None and await _is_marked_fips(database):
            return (
                f"This database was set up with {_SETTING}: true, so the proxy only starts on it in FIPS mode. "
                f"To stop using FIPS mode on it for good, run "
                f"DELETE FROM \"LiteLLM_Config\" WHERE param_name = '{FIPS_MARKER}'."
            )
        return None
    if database is None:
        return f"{_SETTING} is on but the database can't be reached to check it." if database_is_configured else None
    try:
        state: Final = await database.state()
    except Exception as error:  # noqa: BLE001  # a value, so FIPS mode refuses a database it can't check
        return f"{_SETTING} is on but the database could not be checked: {error}"
    if state.holds_rows and not state.marked_fips:
        return (
            f"{_SETTING} is on but this database already holds data written without it. "
            "FIPS mode only starts on a new, empty database."
        )
    return None


async def enforce_fips_database(
    database: FipsRepository | None, *, fips_mode: bool, database_is_configured: bool
) -> None:
    """Raise FipsModeError with the fips_database_refusal reason, if there is one, then mark the database as set up
    in FIPS mode when FIPS mode is on."""
    refusal: Final = await fips_database_refusal(
        database, fips_mode=fips_mode, database_is_configured=database_is_configured
    )
    if refusal is not None:
        raise FipsModeError(refusal)
    if database is not None and fips_mode:
        await database.mark_fips()
