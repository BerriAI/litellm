"""
Auth Checks for Organizations
"""

from typing import Dict, List, Optional, Tuple

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

    1. Only Proxy Admins can access /organization/new
    2. IF route is a LiteLLMRoutes.org_admin_only_routes, then check if user is an Org Admin for that organization

    """

    if user_object is None:
        return

    passed_organization_id: Optional[str] = request_body.get("organization_id", None)

    if route == "/organization/new":
        if user_object.user_role != LitellmUserRoles.PROXY_ADMIN.value:
            raise ProxyException(
                message=f"Only proxy admins can create new organizations. You are {user_object.user_role}",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )

    if user_object.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return

    # Checks if route is an Org Admin Only Route
    if route in LiteLLMRoutes.org_admin_only_routes.value:
        (
            _user_organizations,
            _user_organization_role_mapping,
        ) = get_user_organization_info(user_object)

        if user_object.organization_memberships is None:
            raise ProxyException(
                message=f"Tried to access route={route} but you are not a member of any organization. Please contact the proxy admin to request access.",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        if passed_organization_id is None:
            raise ProxyException(
                message="Passed organization_id is None, please pass an organization_id in your request",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

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
                message=f"You do not have the required role to perform {route} in Organization {passed_organization_id}. Your role is {user_role} in Organization {passed_organization_id}",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )
    elif route == "/team/new":
        # if user is part of multiple teams, then they need to specify the organization_id
        (
            _user_organizations,
            _user_organization_role_mapping,
        ) = get_user_organization_info(user_object)
        if (
            user_object.organization_memberships is not None
            and len(user_object.organization_memberships) > 0
        ):
            if passed_organization_id is None:
                raise ProxyException(
                    message=f"Passed organization_id is None, please specify the organization_id in your request. You are part of multiple organizations: {_user_organizations}",
                    type=ProxyErrorTypes.auth_error.value,
                    param="organization_id",
                    code=status.HTTP_401_UNAUTHORIZED,
                )

            _user_role_in_passed_org = _user_organization_role_mapping.get(
                passed_organization_id
            )
            if _user_role_in_passed_org != LitellmUserRoles.ORG_ADMIN.value:
                raise ProxyException(
                    message=f"You do not have the required role to call {route}. Your role is {_user_role_in_passed_org} in Organization {passed_organization_id}",
                    type=ProxyErrorTypes.auth_error.value,
                    param="user_role",
                    code=status.HTTP_401_UNAUTHORIZED,
                )


def get_user_organization_info(
    user_object: LiteLLM_UserTable,
) -> Tuple[List[str], Dict[str, Optional[LitellmUserRoles]]]:
    """
    Helper function to extract user organization information.

    Args:
        user_object (LiteLLM_UserTable): The user object containing organization memberships.

    Returns:
        Tuple[List[str], Dict[str, Optional[LitellmUserRoles]]]: A tuple containing:
            - List of organization IDs the user is a member of
            - Dictionary mapping organization IDs to user roles
    """
    _user_organizations: List[str] = []
    _user_organization_role_mapping: Dict[str, Optional[LitellmUserRoles]] = {}

    if user_object.organization_memberships is not None:
        for _membership in user_object.organization_memberships:
            if _membership.organization_id is not None:
                _user_organizations.append(_membership.organization_id)
                _user_organization_role_mapping[_membership.organization_id] = _membership.user_role  # type: ignore

    return _user_organizations, _user_organization_role_mapping


def _user_is_org_admin(
    request_data: dict,
    user_object: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    Helper function to check if user is an org admin for the passed organization_id
    """
    if request_data.get("organization_id", None) is None:
        return False

    if user_object is None:
        return False

    if user_object.organization_memberships is None:
        return False

    for _membership in user_object.organization_memberships:
        if _membership.organization_id == request_data.get("organization_id", None):
            if _membership.user_role == LitellmUserRoles.ORG_ADMIN.value:
                return True

    return False
