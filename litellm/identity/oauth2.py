"""OAuth2 identity construction: introspection response -> ``UserAPIKeyAuth``.

The HTTP plumbing stays in ``oauth2_check.Oauth2Handler``; this module maps
an already-validated response payload into the carrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


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
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

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
            raise ValueError(f"Invalid OAuth2 role: {raw_role!r}") from e

    return UserAPIKeyAuth(
        api_key=token,
        team_id=user_team_id,
        user_id=user_id,
        user_role=user_role,
    )
