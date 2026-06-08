"""OAuth2 identity construction.

Owns the translation from an OAuth2 introspection / userinfo response
into the proxy's carrier model ``UserAPIKeyAuth``. The HTTP plumbing —
introspection endpoint detection, request signing, error handling —
stays in ``litellm.proxy.auth.oauth2_check.Oauth2Handler``; this module
just maps a validated response payload into the carrier so the JWT and
OAuth2 paths converge on the same construction surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

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
    user_role: Optional[str] = response_data.get(user_role_field_name)
    user_team_id: Optional[str] = response_data.get(user_team_id_field_name)

    return UserAPIKeyAuth(
        api_key=token,
        team_id=user_team_id,
        user_id=user_id,
        user_role=cast("LitellmUserRoles", user_role),
    )
