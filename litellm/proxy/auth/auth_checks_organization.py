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

    1. If route is a /new or /generate route ensure organization is passed in request and user is part of it
        1a Check UserRole in organization and ensure user has Create permissions


    Raises Exception:
        - when a user is part of an organization and tries creating a team / user without specifying an organization_id
        - when a user does not have the appropriate permissions within their organization
    """

    if user_object is None:
        return

    if (
        user_object.organization_memberships is None
        or len(user_object.organization_memberships) == 0
    ):
        return

    _user_organizations: List[str] = []
    _user_organization_role_mapping: Dict[str, Optional[LitellmUserRoles]] = {}
    for _membership in user_object.organization_memberships:
        if _membership.organization_id is not None:
            _user_organizations.append(_membership.organization_id)
            _user_organization_role_mapping[_membership.organization_id] = _membership.user_role  # type: ignore

    if route in LiteLLMRoutes.management_create_routes.value:
        if "organization_id" not in request_body:
            raise Exception(
                "Please specify 'organization_id' in request body, you are part of multiple_organizations"
            )

        passed_organization_id = request_body["organization_id"]

        if passed_organization_id not in _user_organizations:
            raise ProxyException(
                message=f"You are not a member of the organization specified in the request body. Passed organization_id: {passed_organization_id}",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        user_role: Optional[LitellmUserRoles] = _user_organization_role_mapping.get(
            passed_organization_id
        )
        if user_role is None:
            raise ProxyException(
                message=f"You do not have a role within the selected organization. Passed organization_id: {passed_organization_id}",
                type=ProxyErrorTypes.auth_error.value,
                param="organization_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )
        elif (
            user_role == LitellmUserRoles.INTERNAL_USER.value
            and route not in LiteLLMRoutes.internal_user_routes.value
        ):
            raise ProxyException(
                message=f"You do not have the correct role to access this route in organization_id: {passed_organization_id}, role: {user_role}. Tried to call route: {route}",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )
        elif (
            user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value
            and route not in LiteLLMRoutes.internal_user_view_only_routes.value
        ):
            raise ProxyException(
                message=f"You do not have the correct role to access this route in organization_id: {passed_organization_id}, role: {user_role}. Tried to call route: {route}",
                type=ProxyErrorTypes.auth_error.value,
                param="user_role",
                code=status.HTTP_401_UNAUTHORIZED,
            )
