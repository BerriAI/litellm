"""Password-strength policy enforcement for locally-managed proxy users.

Applied at every path that persists a new or changed password for a DB-backed
user (``/user/update``, ``/user/bulk_update``, and the invitation onboarding
claim flow), so the strength bar is configured in one place instead of
per-endpoint.

Also screens new passwords against known data breaches via the
haveibeenpwned.com (HIBP) k-anonymity range API: only the first 5 characters
of the password's SHA-1 hash ever leave the proxy, and the check fails open
(allows the password) when HIBP is unreachable.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm._version import version
from litellm.constants import HIBP_RANGE_API_BASE
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.types.llms.custom_http import httpxSpecialProvider

HIBP_TIMEOUT_SECONDS: Final = 5.0

DEFAULT_MIN_LENGTH: Final = 12
MIN_ALLOWED_LENGTH: Final = 8


def _has_uppercase(password: str) -> bool:
    return any(ch.isupper() for ch in password)


def _has_lowercase(password: str) -> bool:
    return any(ch.islower() for ch in password)


def _has_digit(password: str) -> bool:
    return any(ch.isdigit() for ch in password)


def _has_special_character(password: str) -> bool:
    """Unicode-aware: a letter or digit from ANY script counts as
    alphanumeric, not just ASCII, so an accented letter (e.g. the second
    character of "Passwörd1234") cannot be miscounted as the required
    special character the way an ASCII-only `[^A-Za-z0-9]` regex would."""
    return any(not ch.isalnum() for ch in password)


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    min_length: int
    require_uppercase: bool
    require_lowercase: bool
    require_numbers: bool
    require_special_characters: bool


def _configured_min_length(general_settings: Mapping[str, object]) -> int:
    """The configured minimum, floored at MIN_ALLOWED_LENGTH so a nonpositive
    or too-low override (a typo, or `0`/`false` coercing through) cannot
    silently disable the length requirement rather than merely relaxing it."""
    min_length_setting: Final = general_settings.get("password_policy_min_length")
    if isinstance(min_length_setting, bool) or not isinstance(min_length_setting, (int, float)):
        return DEFAULT_MIN_LENGTH
    return max(MIN_ALLOWED_LENGTH, int(min_length_setting))


def get_password_policy(general_settings: Mapping[str, object]) -> PasswordPolicy:
    return PasswordPolicy(
        min_length=_configured_min_length(general_settings),
        require_uppercase=general_settings.get("password_policy_require_uppercase", True) is not False,
        require_lowercase=general_settings.get("password_policy_require_lowercase", True) is not False,
        require_numbers=general_settings.get("password_policy_require_numbers", True) is not False,
        require_special_characters=(
            general_settings.get("password_policy_require_special_characters", True) is not False
        ),
    )


def _policy_violations(password: str, policy: PasswordPolicy) -> tuple[str, ...]:
    checks: Final = (
        (len(password) < policy.min_length, f"be at least {policy.min_length} characters long"),
        (policy.require_uppercase and not _has_uppercase(password), "include an uppercase letter"),
        (policy.require_lowercase and not _has_lowercase(password), "include a lowercase letter"),
        (policy.require_numbers and not _has_digit(password), "include a number"),
        (policy.require_special_characters and not _has_special_character(password), "include a special character"),
    )
    return tuple(message for failed, message in checks if failed)


def validate_password_policy(password: str, general_settings: Mapping[str, object]) -> None:
    """Raise ``ProxyException`` (400) if ``password`` fails the configured policy."""
    policy: Final = get_password_policy(general_settings)
    violations: Final = _policy_violations(password, policy)
    if not violations:
        return
    raise ProxyException(
        message="Password does not meet the required policy: must " + ", ".join(violations) + ".",
        type=ProxyErrorTypes.validation_error,
        param="password",
        code=400,
    )


def _hibp_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PasswordBreachCheck,
        params={"timeout": HIBP_TIMEOUT_SECONDS},
    )


def _is_suffix_in_range_response(response_body: str, hash_suffix: str) -> bool:
    for line in response_body.upper().splitlines():
        entry_suffix, _, count = line.strip().partition(":")
        if entry_suffix == hash_suffix:
            return int(count.strip() or "0") > 0
    return False


async def _is_password_breached(password: str, client: AsyncHTTPHandler) -> bool:
    # usedforsecurity=False: SHA-1 is only a lookup key into the HIBP dataset, so no security property rests on it
    sha1_hex: Final = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
    try:
        response: Final = await client.get(
            f"{HIBP_RANGE_API_BASE}/{sha1_hex[:5]}",
            headers={"Add-Padding": "true", "User-Agent": f"litellm-proxy/{version}"},
        )
        response.raise_for_status()
        breached: Final = _is_suffix_in_range_response(response.text, sha1_hex[5:])
    except Exception as e:
        verbose_proxy_logger.warning("Breached-password check skipped, HIBP lookup failed: %s", e)
        return False
    return breached


async def validate_password_not_breached(
    password: str,
    general_settings: Mapping[str, object],
    client: AsyncHTTPHandler | None = None,
) -> None:
    """Raise ``ProxyException`` (400) if ``password`` appears in a known data breach.

    Fails open: an unreachable or misbehaving HIBP allows the password."""
    check_enabled: Final = general_settings.get("password_policy_check_breached_passwords", True) is not False
    if not check_enabled:
        return
    if not await _is_password_breached(password, client if client is not None else _hibp_client()):
        return
    raise ProxyException(
        message=(
            "This password appears in known data breaches and cannot be used. Please choose a different password."
        ),
        type=ProxyErrorTypes.validation_error,
        param="password",
        code=400,
    )
