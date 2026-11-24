"""
Types for the management endpoints

Might include fastapi/proxy requirements.txt related imports
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Final, Protocol, cast

import email_validator
from fastapi_sso.sso.base import OpenID
from pydantic import field_validator

from litellm.proxy._types import LitellmUserRoles

# Ordered highest to lowest privilege
LITELLM_USER_ROLE_HIERARCHY: Final = (
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
)


def highest_privilege_role(roles: Iterable[LitellmUserRoles]) -> LitellmUserRoles | None:
    """
    Pick the highest privilege role out of the roles an IdP asserted for one user.

    IdPs do not guarantee ordering within a multi-valued role claim, so a user holding
    several roles resolves to the most privileged one rather than whichever came first.
    Roles the hierarchy does not rank (org_admin, team, customer) resolve by name to stay
    deterministic.

    Args:
        roles: The roles resolved from the claim

    Returns:
        The highest privilege role, or None if `roles` is empty
    """
    resolved: Final = frozenset(roles)
    if not resolved:
        return None

    ranked: Final = next((role for role in LITELLM_USER_ROLE_HIERARCHY if role in resolved), None)
    return ranked if ranked is not None else min(resolved, key=lambda role: role.value)


def is_valid_litellm_user_role(role_str: str) -> bool:
    """
    Check if a string is a valid LitellmUserRoles enum value (case-insensitive).

    Args:
        role_str: String to validate (e.g., "proxy_admin", "PROXY_ADMIN", "internal_user")

    Returns:
        True if the string matches a valid LitellmUserRoles value, False otherwise
    """
    try:
        # Use _value2member_map_ for O(1) lookup, case-insensitive
        return role_str.lower() in LitellmUserRoles._value2member_map_
    except Exception:
        return False


def _role_from_claim_value(role_str: object) -> LitellmUserRoles | None:
    if not isinstance(role_str, str):
        return None
    # Use _value2member_map_ for O(1) lookup, case-insensitive
    result: Final = LitellmUserRoles._value2member_map_.get(role_str.lower())
    return cast(LitellmUserRoles | None, result)


def get_litellm_user_role(role_str: object) -> LitellmUserRoles | None:
    """
    Convert a string (or list of strings) to a LitellmUserRoles enum if valid (case-insensitive).

    Handles list inputs since some SSO providers (e.g., Keycloak) return roles
    as arrays like ["proxy_admin"] instead of plain strings. A claim carrying several
    roles resolves to the highest privilege one, so a user does not lose access just
    because the IdP listed a weaker role first.

    Args:
        role_str: String or list to convert (e.g., "proxy_admin", ["proxy_admin"])

    Returns:
        LitellmUserRoles enum if valid, None otherwise
    """
    if isinstance(role_str, (list, tuple)):
        entries: Final = cast(Sequence[object], role_str)  # cast-ok: isinstance narrows the claim, not its elements
        return highest_privilege_role(
            role for role in (_role_from_claim_value(entry) for entry in entries) if role is not None
        )
    return _role_from_claim_value(role_str)


class CustomOpenID(OpenID):
    team_ids: list[str]
    team_id_to_upsert: str | None = None
    user_role: LitellmUserRoles | None = None
    extra_fields: dict[str, object] | None = None
    email_verified: bool | None = None


class _EmailValidator(Protocol):
    @property
    def SPECIAL_USE_DOMAIN_NAMES(self) -> Sequence[str]: ...


_EMAIL_VALIDATOR: Final = cast(_EmailValidator, email_validator)


class GenericOpenID(CustomOpenID):
    @field_validator("email", mode="wrap")
    @classmethod
    def _validate_directory_email(cls, value: object, validate: Callable[[object], str | None]) -> str | None:
        if not isinstance(value, str):
            return validate(value)
        local, _, domain = value.partition("@")
        if "@" in domain or value != value.strip() or any(char in value for char in "<>"):
            return validate(value)

        lower_domain: Final = domain.lower()
        special_use_name: Final = max(
            (
                name
                for name in _EMAIL_VALIDATOR.SPECIAL_USE_DOMAIN_NAMES
                if lower_domain == name or lower_domain.endswith("." + name)
            ),
            key=len,
            default=None,
        )
        if special_use_name is None:
            return validate(value)

        validation_suffix: Final = ".".join("a" * len(label) for label in special_use_name.split("."))
        domain_prefix: Final = domain[: -len(special_use_name)]
        normalized: Final = validate(f"{local}@{domain_prefix}{validation_suffix}")
        if normalized is None or not normalized.endswith(validation_suffix):
            raise ValueError("Invalid email domain")
        return normalized.removesuffix(validation_suffix) + special_use_name
