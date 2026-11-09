"""
Auth Checks for Organizations
"""

from collections.abc import Awaitable, Callable
from typing import Final

from litellm.proxy._types import *


def organization_role_based_access_check(
    request_body: dict,
    user_object: LiteLLM_UserTable | None,
    route: str,
):
    from litellm.proxy.auth import org_route_adapter

    return org_route_adapter.organization_role_based_access_check(
        request_body=request_body, user_object=user_object, route=route
    )


def get_user_organization_info(
    user_object: LiteLLM_UserTable,
) -> tuple[list[str], dict[str, LitellmUserRoles | None]]:
    """
    Helper function to extract user organization information.

    Args:
        user_object (LiteLLM_UserTable): The user object containing organization memberships.

    Returns:
        Tuple[List[str], Dict[str, Optional[LitellmUserRoles]]]: A tuple containing:
            - List of organization IDs the user is a member of
            - Dictionary mapping organization IDs to user roles
    """
    _user_organizations: Final[list[str]] = []
    _user_organization_role_mapping: Final[dict[str, LitellmUserRoles | None]] = {}

    if user_object.organization_memberships is not None:
        for _membership in user_object.organization_memberships:
            if _membership.organization_id is not None:
                _user_organizations.append(_membership.organization_id)
                _user_organization_role_mapping[_membership.organization_id] = _membership.user_role

    return _user_organizations, _user_organization_role_mapping


def user_is_org_admin(
    request_data: dict,
    user_object: LiteLLM_UserTable | None = None,
) -> bool:
    from litellm.proxy.auth import org_route_adapter

    return org_route_adapter.user_is_org_admin(request_data=request_data, user_object=user_object)


_user_is_org_admin: Final = user_is_org_admin


TEAM_ORG_CONTEXT_ROUTES: Final = frozenset({"/team/update"})
# The RESTful update route carries the team id in the path. Match on the route
# template so the sibling /team/<verb> routes (which share the single-segment
# shape) are not mistaken for it and don't trigger a team lookup.
PATCH_TEAM_ROUTE_TEMPLATE: Final = "/team/{team_id}"


async def add_team_org_context_to_request_body(
    route: str,
    request_body: dict,
    fetch_team_org_id: Callable[[str], Awaitable[str | None]],
    route_template: str | None = None,
) -> dict:
    """
    Return a copy of request_body with organization_id resolved from the target
    team when the route identifies the team by team_id and the caller did not
    pass organization_id. This lets an org admin of the team's own org reach the
    org-scoped branch of the route gate (which keys off organization_id) without
    the client having to send it. Returns request_body unchanged when it does
    not apply, so callers that already pass organization_id and non-team routes
    are untouched.

    The team_id is taken from the body for TEAM_ORG_CONTEXT_ROUTES, or from the
    last path segment when ``route_template`` is the ``/team/{team_id}`` route.
    """
    if request_body.get("organization_id"):
        return request_body

    if route in TEAM_ORG_CONTEXT_ROUTES:
        team_id: str | None = request_body.get("team_id")
    elif route_template == PATCH_TEAM_ROUTE_TEMPLATE:
        team_id = route.rsplit("/", 1)[-1]
    else:
        return request_body

    if not isinstance(team_id, str) or not team_id:
        return request_body
    org_id: Final = await fetch_team_org_id(team_id)
    if not org_id:
        return request_body
    return {**request_body, "organization_id": org_id}
