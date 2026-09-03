"""Password-strength policy enforcement for locally-managed proxy users.

Applied at every path that persists a new or changed password for a DB-backed
user (``/user/update``, ``/user/bulk_update``, and the invitation onboarding
claim flow), so the strength bar is configured in one place instead of
per-endpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.proxy._types import ProxyErrorTypes, ProxyException

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
