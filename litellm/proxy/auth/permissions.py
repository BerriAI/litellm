"""
Granular permission strings for the LiteLLM proxy.

Permission format: `resource:action`

This module defines the valid permission strings that can be granted to
team members via `extra_permissions`. It is the first step toward a full
Permission Strings RBAC system (custom roles, org intersection, denied_permissions).

New resource permissions should be added here as enums and included in
VALID_PERMISSIONS so that the validation in team_member_update rejects typos.
"""

from enum import Enum
from typing import Dict, List


class MCPPermission(str, Enum):
    """Permissions for MCP server management."""

    READ = "mcp:read"
    CREATE = "mcp:create"
    UPDATE = "mcp:update"
    DELETE = "mcp:delete"


# All known permission strings — used for validation when granting permissions.
# Grows as more resource permissions are added (keys:create, teams:create, etc.)
VALID_PERMISSIONS: set = {p.value for p in MCPPermission}


def get_available_permissions() -> List[Dict[str, str]]:
    """
    Return the list of valid permission strings with labels, grouped by resource.

    Used by the GET /team/available_permissions endpoint for UI dropdowns.
    """
    return [
        {
            "value": MCPPermission.READ.value,
            "label": "View MCP servers",
            "resource": "mcp",
        },
        {
            "value": MCPPermission.CREATE.value,
            "label": "Create MCP servers",
            "resource": "mcp",
        },
        {
            "value": MCPPermission.UPDATE.value,
            "label": "Edit MCP servers",
            "resource": "mcp",
        },
        {
            "value": MCPPermission.DELETE.value,
            "label": "Delete MCP servers",
            "resource": "mcp",
        },
    ]
