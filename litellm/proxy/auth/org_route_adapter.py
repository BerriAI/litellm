from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm._auth.org_routes import (
    OrganizationMembership,
    OrganizationRouteFacts,
    organization_route,
)
from litellm._auth.org_routes import (
    user_is_org_admin as native_org_admin,
)
from litellm.proxy._types import LiteLLM_UserTable, LiteLLMRoutes, ProxyErrorTypes, ProxyException

_STRINGS: Final = TypeAdapter(tuple[str, ...])


def memberships(user: LiteLLM_UserTable | None) -> tuple[OrganizationMembership, ...] | None:
    return (
        tuple(OrganizationMembership(item.organization_id, item.user_role) for item in user.organization_memberships)
        if user is not None and user.organization_memberships is not None
        else None
    )


def user_is_org_admin(request_data: Mapping[str, object], user_object: LiteLLM_UserTable | None = None) -> bool:
    singular: Final = request_data.get("organization_id")
    plural: Final = request_data.get("organizations")
    return native_org_admin(
        memberships(user_object) or (),
        (
            *((singular,) if isinstance(singular, str) else ()),
            *(_STRINGS.validate_python(plural) if isinstance(plural, list) else ()),
        ),
    )


def organization_role_based_access_check(
    request_body: Mapping[str, object], user_object: LiteLLM_UserTable | None, route: str
) -> None:
    passed: Final = request_body.get("organization_id")
    organization_id: Final = passed if isinstance(passed, str) else None
    role: Final = user_object.user_role if user_object is not None else None
    decision: Final = organization_route(
        OrganizationRouteFacts(
            route,
            user_object is not None,
            role,
            memberships(user_object),
            organization_id,
            tuple(LiteLLMRoutes.org_admin_only_routes.value),
        )
    )
    messages: Final = {
        "CreateOrganization": (f"Only proxy admins can create new organizations. You are {role}", "user_role"),
        "NoMemberships": (
            f"Tried to access route={route} but you are not a member of any organization. Please contact the proxy admin to request access.",
            "organization_id",
        ),
        "MissingOrganization": (
            "Passed organization_id is None, please pass an organization_id in your request",
            "organization_id",
        ),
        "MissingRole": (
            f"You do not have a role within the selected organization. Passed organization_id: {organization_id}. Please contact the organization admin to request access.",
            "organization_id",
        ),
        "WrongRole": (
            f"You do not have the required role to perform {route} in Organization {organization_id}. Your role is {decision.organization_role} in Organization {organization_id}",
            "user_role",
        ),
        "TeamMissingOrganization": (
            f"Passed organization_id is None, please specify the organization_id in your request. You are part of multiple organizations: {list(decision.organization_ids)}",
            "organization_id",
        ),
        "TeamWrongRole": (
            f"You do not have the required role to call {route}. Your role is {decision.organization_role} in Organization {organization_id}",
            "user_role",
        ),
    }
    if decision.denial is not None:
        message, param = messages[decision.denial]
        raise ProxyException(message=message, type=ProxyErrorTypes.auth_error.value, param=param, code=401)
