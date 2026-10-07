import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from typing_extensions import assert_never

from litellm.secret_managers.main import str_to_bool

FIPS_MODE_ENV_VAR: Final = "LITELLM_FIPS_MODE"
SSL_VERIFY_ENV_VAR: Final = "SSL_VERIFY"
SSL_VERIFY_SETTING: Final = "litellm_settings.ssl_verify"
REFUSAL_PREFIX: Final = "LiteLLM proxy refused to start"

_TRUE_VALUES: Final = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True, slots=True)
class FipsModeOff:
    pass


@dataclass(frozen=True, slots=True)
class FipsModeOn:
    pass


@dataclass(frozen=True, slots=True)
class MalformedFipsMode:
    value: str


FipsModeSetting = FipsModeOff | FipsModeOn | MalformedFipsMode


@dataclass(frozen=True, slots=True)
class ProviderDoesNotEnforceFips:
    pass


@dataclass(frozen=True, slots=True)
class TlsVerificationDisabled:
    sources: tuple[str, ...]


FipsBootRefusal = MalformedFipsMode | ProviderDoesNotEnforceFips | TlsVerificationDisabled
FipsBootVerdict = FipsModeOff | FipsModeOn | FipsBootRefusal


class FipsModeError(Exception):
    pass


def parse_fips_mode(raw: str | None) -> FipsModeSetting:
    if raw is None:
        return FipsModeOff()
    normalized: Final = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return FipsModeOn()
    if normalized in _FALSE_VALUES:
        return FipsModeOff()
    return MalformedFipsMode(value=raw)


def is_fips_mode(environ: Callable[[str], str | None] = os.environ.get) -> bool:
    return isinstance(parse_fips_mode(environ(FIPS_MODE_ENV_VAR)), FipsModeOn)


def openssl_enforces_fips() -> bool:
    """MD5 is not an approved digest, so an enforcing FIPS provider refuses it even when asked for security use."""
    try:
        hashlib.md5(b"", usedforsecurity=True)
    except ValueError:
        return True
    return False


def fips_boot_verdict(
    *,
    raw_fips_mode: str | None,
    provider_enforces_fips: Callable[[], bool],
    ssl_verify_environment: str | None,
    ssl_verify_setting: object,
) -> FipsBootVerdict:
    setting: Final = parse_fips_mode(raw_fips_mode)
    match setting:
        case FipsModeOff() | MalformedFipsMode():
            return setting
        case FipsModeOn():
            pass
        case _:
            assert_never(setting)
    disabled: Final = tuple(
        source
        for source, off in (
            (SSL_VERIFY_ENV_VAR, _is_off(ssl_verify_environment)),
            (SSL_VERIFY_SETTING, _is_off(ssl_verify_setting)),
        )
        if off
    )
    if disabled:
        return TlsVerificationDisabled(sources=disabled)
    if not provider_enforces_fips():
        return ProviderDoesNotEnforceFips()
    return setting


def enforce_fips_boot_verdict(verdict: FipsBootVerdict, announce: Callable[[str], object]) -> None:
    match verdict:
        case FipsModeOff() | FipsModeOn():
            return
        case MalformedFipsMode() | ProviderDoesNotEnforceFips() | TlsVerificationDisabled():
            message: Final = render_refusal(verdict)
            announce(f"\n{message}\n\n")
            raise FipsModeError(message)
        case _:
            assert_never(verdict)


def render_refusal(refusal: FipsBootRefusal) -> str:
    match refusal:
        case MalformedFipsMode():
            return (
                f"{REFUSAL_PREFIX}: {FIPS_MODE_ENV_VAR}={refusal.value} is not a boolean.\n"
                f"Set {FIPS_MODE_ENV_VAR} to true or false, or unset it."
            )
        case ProviderDoesNotEnforceFips():
            return (
                f"{REFUSAL_PREFIX}: {FIPS_MODE_ENV_VAR} is on but this Python does not enforce FIPS.\n"
                "Its OpenSSL still allows non-approved algorithms (MD5 succeeded), so passwords and keys would be\n"
                "protected with algorithms the FIPS 140-3 policy forbids. Run the proxy from a FIPS image whose\n"
                f"OpenSSL FIPS provider is enabled, or unset {FIPS_MODE_ENV_VAR} on a non-FIPS runtime."
            )
        case TlsVerificationDisabled():
            return (
                f"{REFUSAL_PREFIX}: {FIPS_MODE_ENV_VAR} is on but TLS certificate verification is disabled by "
                f"{' and '.join(refusal.sources)}.\nFIPS deployments must verify upstream certificates, so remove the "
                "override or point ssl_verify at a CA bundle instead."
            )
    return assert_never(refusal)


def _is_off(value: object) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return str_to_bool(value) is False
    return False
