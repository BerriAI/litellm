"""
Auth Checks for Organizations, decided by gateway-auth-core.
"""

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Final

from litellm.proxy._types import LiteLLM_UserTable
from litellm.rust_bridge._native import auth_org_context_team, auth_organization_denial, auth_user_is_org_admin


def organization_role_based_access_check(
    request_body: dict,
    user_object: LiteLLM_UserTable | None,
    route: str,
) -> None:
    """Only proxy admins create organizations, and an org admin only route needs the org admin role in the
    organization the request names."""
    from litellm.proxy.auth.denial import denial_error  # noqa: PLC0415  # denial imports auth_checks
    from litellm.proxy.auth.route_checks import route_facts  # noqa: PLC0415  # route_checks imports this module

    denial: Final = auth_organization_denial(route_facts(route, None, request_body, None, user_object))
    if denial is not None:
        raise denial_error(denial)


def user_is_org_admin(
    request_data: dict,
    user_object: LiteLLM_UserTable | None = None,
) -> bool:
    """An org admin of every organization the request names in `organization_id` and `organizations`."""
    from litellm.proxy.auth.route_checks import route_facts  # noqa: PLC0415  # route_checks imports this module

    return auth_user_is_org_admin(route_facts("", None, request_data, None, user_object))


_user_is_org_admin: Final = user_is_org_admin


async def add_team_org_context_to_request_body(
    route: str,
    request_body: dict,
    fetch_team_org_id: Callable[[str], Awaitable[str | None]],
    route_template: str | None = None,
) -> dict:
    """
    Return a copy of request_body with organization_id resolved from the target team when the route identifies
    the team by team_id and the caller did not pass organization_id, so an org admin of the team's own org
    reaches the org-scoped branch of the route gate. Returns request_body unchanged when it does not apply.
    """
    from litellm.proxy.auth.route_checks import route_facts  # noqa: PLC0415  # route_checks imports this module

    team_id: Final = auth_org_context_team(
        dataclasses.replace(route_facts(route, None, request_body, None, None), route_template=route_template)
    )
    if team_id is None:
        return request_body
    org_id: Final = await fetch_team_org_id(team_id)
    if not org_id:
        return request_body
    return {**request_body, "organization_id": org_id}
