"""
Organization Auth Checks with Enhanced Security and Type Safety
"""

from uuid import UUID
from typing import Dict, List, Optional, Tuple
from fastapi import status
from litellm.proxy._types import *


def organization_role_based_access_check(
    request_body: dict,
    user_object: Optional[LiteLLM_UserTable],
    route: str,
) -> None:
    """
    Enhanced organization access control with:
    - UUID validation for organization_id
    - Type-safe role comparisons
    - Consolidated permission checks
    """
    if user_object is None:
        return

    # Validate organization_id format if present
    passed_organization_id = request_body.get("organization_id")
    if passed_organization_id:
        try:
            UUID(passed_organization_id, version=4)
        except ValueError as e:
            raise ProxyException(
                message=f"Invalid organization_id format: {str(e)}",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_400_BAD_REQUEST,
            )

    # Proxy admin-only route check
    if route == "/organization/new":
        if user_object.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise ProxyException(
                message="Insufficient permissions for organization creation",
                detail=f"Required role: {LitellmUserRoles.PROXY_ADMIN.value}",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
            )
        return

    # Bypass checks for proxy admins
    if user_object.user_role == LitellmUserRoles.PROXY_ADMIN:
        return

    # Get organization info once
    user_orgs, role_mapping = get_user_organization_info(user_object)

    # Org admin required routes
    if route in LiteLLMRoutes.org_admin_only_routes.value:
        if not passed_organization_id:
            raise ProxyException(
                message="organization_id required for this operation",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_400_BAD_REQUEST,
            )

        user_role = role_mapping.get(passed_organization_id, LitellmUserRoles.INTERNAL_USER)
        
        if user_role != LitellmUserRoles.ORG_ADMIN:
            available_orgs = "\n".join([f"{k}: {v.value}" for k,v in role_mapping.items()])
            raise ProxyException(
                message="Organization admin privileges required",
                detail=f"Required role: {LitellmUserRoles.ORG_ADMIN.value}\nYour roles:\n{available_orgs}",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
            )


def get_user_organization_info(
    user_object: LiteLLM_UserTable,
) -> Tuple[List[str], Dict[str, LitellmUserRoles]]:
    """
    Returns validated organization info with:
    - Type-safe role conversions
    - Empty collection handling
    """
    if not user_object or not user_object.organization_memberships:
        return [], {}

    organizations = []
    role_mapping = {}

    for membership in user_object.organization_memberships:
        if not membership.organization_id:
            continue

        organizations.append(membership.organization_id)
        
        # Convert to enum with fallback
        try:
            role = LitellmUserRoles(membership.user_role)
        except ValueError:
            role = LitellmUserRoles.INTERNAL_USER
            
        role_mapping[membership.organization_id] = role

    return organizations, role_mapping


def _user_is_org_admin(
    request_data: dict,
    user_object: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    Efficient admin check with:
    - Early exit conditions
    - Generator expression for performance
    """
    org_id = request_data.get("organization_id")
    if not org_id or not user_object:
        return False

    return any(
        m.organization_id == org_id 
        and m.user_role == LitellmUserRoles.ORG_ADMIN
        for m in user_object.organization_memberships or []
    )
