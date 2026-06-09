"""OAuth2 identity construction: introspection response -> ``UserAPIKeyAuth``.

The HTTP plumbing stays in ``oauth2_check.Oauth2Handler``; this module maps
an already-validated response payload into the carrier.
"""

from __future__ import annotations

import os
from typing import Optional

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

UNKNOWN_ROLE_DEFAULT_ENV = "LITELLM_OAUTH2_UNKNOWN_ROLE_DEFAULT"


def _unknown_role_fallback() -> Optional[LitellmUserRoles]:
    raw = os.getenv(UNKNOWN_ROLE_DEFAULT_ENV)
    if not raw:
        return None
    return LitellmUserRoles(raw)


def build_user_api_key_auth_from_oauth2_response(
    *,
    token: str,
    response_data: dict,
    user_id_field_name: str = "sub",
    user_role_field_name: str = "role",
    user_team_id_field_name: str = "team_id",
) -> "UserAPIKeyAuth":
    """Build a ``UserAPIKeyAuth`` carrier from an OAuth2 response.

    ``response_data`` is the parsed JSON body of an OAuth2 introspection
    or userinfo response. The three ``*_field_name`` parameters let
    deployments map their IdP's claim names onto our canonical fields;
    defaults match the OAuth2 introspection spec (``sub``) plus
    ``role`` / ``team_id`` for the role and team claims.

    Active-token validation, scope checks, and token-not-active rejection
    happen upstream in ``Oauth2Handler``; this builder trusts its input.

    Unknown IdP role values are rejected by default. Set
    ``LITELLM_OAUTH2_UNKNOWN_ROLE_DEFAULT`` to a valid
    ``LitellmUserRoles`` value to map unknown roles to that fallback;
    an invalid env value raises on first use so misconfiguration is loud.
    """
    user_id: Optional[str] = response_data.get(user_id_field_name)
    raw_role = response_data.get(user_role_field_name)
    user_team_id: Optional[str] = response_data.get(user_team_id_field_name)

    user_role: Optional[LitellmUserRoles]
    if raw_role is None:
        user_role = None
    else:
        try:
            user_role = LitellmUserRoles(raw_role)
        except ValueError as e:
            fallback = _unknown_role_fallback()
            if fallback is not None:
                user_role = fallback
            else:
                raise ValueError(f"Invalid OAuth2 role: {raw_role!r}") from e

    return UserAPIKeyAuth(
        api_key=token,
        team_id=user_team_id,
        user_id=user_id,
        user_role=user_role,
    )
