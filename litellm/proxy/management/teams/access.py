"""Moved to ``authz``. Kept because the published litellm-enterprise 0.1.73 wheel imports ``is_team_admin`` from here;
delete once the enterprise pin moves to a release that imports from ``authz``."""

from litellm.proxy.management.teams.authz import (
    TEAM_ADMIN_ONLY,
    TEAM_OR_ORG_ADMIN,
    OrgRoles,
    TeamAccess,
    TeamRole,
    is_team_admin,
    team_access_denied,
)

__all__ = [
    "TEAM_ADMIN_ONLY",
    "TEAM_OR_ORG_ADMIN",
    "OrgRoles",
    "TeamAccess",
    "TeamRole",
    "is_team_admin",
    "team_access_denied",
]
