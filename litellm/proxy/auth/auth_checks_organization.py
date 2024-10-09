"""
Auth Checks for Organizations
"""

from fastapi import status

from litellm.proxy._types import *


def organization_role_based_access_check(
    request_body: dict,
    user_object: Optional[LiteLLM_UserTable],
    route: str,
):
    """
    Role based access control checks only run if a user is part of an Organization

    Organization Checks:
    ONLY RUN IF user_object.organization_memberships is not None

    1. Only Proxy Admins and Org Admins can access /organization/* routes
    2. IF user_object.organization_memberships is not None /team/new is called

    """

    if user_object is None:
        return

    passed_organization_id: str = request_body.get("organization_id", None)

    if route == "/organization/new":
        if user_object.user_role != LitellmUserRoles.PROXY_ADMIN.value:
            raise ProxyException(
                message=f"Only proxy admins can create new organizations",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )

    if user_object.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    # Checks if route is an Org Admin Only Route
    if route in LiteLLMRoutes.org_admin_only_routes.value:
        _user_organizations: List[str] = []
        _user_organization_role_mapping: Dict[str, Optional[LitellmUserRoles]] = {}

        if user_object.organization_memberships is None:
            raise ProxyException(
                message=f"Tried to access route={route} but you are not a member of any organization. Please contact the proxy admin to request access.",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        if passed_organization_id is None:
            raise ProxyException(
                message=f"Passed organization_id is None, please pass an organization_id",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        for _membership in user_object.organization_memberships:
            if _membership.organization_id is not None:
                _user_organizations.append(_membership.organization_id)
                _user_organization_role_mapping[_membership.organization_id] = _membership.user_role  # type: ignore

        user_role: Optional[LitellmUserRoles] = _user_organization_role_mapping.get(
            passed_organization_id
        )
        if user_role is None:
            raise ProxyException(
                message=f"You do not have a role within the selected organization. Passed organization_id: {passed_organization_id}. Please contact the organization admin to request access.",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        if user_role != LitellmUserRoles.ORG_ADMIN.value:
            raise ProxyException(
                message=f"You do not have the required role to perform this action. Your role is {user_role} in Organization {passed_organization_id}",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )
