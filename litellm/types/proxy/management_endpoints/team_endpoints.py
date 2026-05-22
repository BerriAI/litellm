from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from litellm.proxy._types import (
    KeyManagementRoutes,
    LiteLLM_DeletedTeamTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
)


class GetTeamMemberPermissionsRequest(BaseModel):
    """Request to get the team member permissions for a team"""

    team_id: str


class GetTeamMemberPermissionsResponse(BaseModel):
    """Response to get the team member permissions for a team"""

    team_id: str
    """
    The team id that the permissions are for
    """

    team_member_permissions: Optional[List[str]] = []
    """
    The team member permissions currently set for the team
    """

    all_available_permissions: List[str]
    """
    All available team member permissions
    """


class UpdateTeamMemberPermissionsRequest(BaseModel):
    """Request to update the team member permissions for a team"""

    team_id: str
    team_member_permissions: List[str]


class BulkUpdateTeamMemberPermissionsRequest(BaseModel):
    """Request to bulk-update team member permissions across teams."""

    permissions: List[KeyManagementRoutes]
    """Permissions to append to the target teams (duplicates are skipped)."""

    team_ids: Optional[List[str]] = None
    """Specific team IDs to update. Required unless apply_to_all_teams is True."""

    apply_to_all_teams: bool = False
    """When True, update all teams. Mutually exclusive with team_ids."""


class BulkUpdateTeamMemberPermissionsResponse(BaseModel):
    """Response for bulk team member permissions update."""

    message: str
    teams_updated: int
    permissions_appended: Optional[List[str]] = None


class TeamListItem(LiteLLM_TeamTable):
    """A team item in the paginated list response, enriched with computed fields."""

    members_count: int = 0
    # Resources inherited from access groups (separate from direct assignments)
    access_group_models: Optional[List[str]] = None
    access_group_mcp_server_ids: Optional[List[str]] = None
    access_group_agent_ids: Optional[List[str]] = None


class TeamListResponse(BaseModel):
    """Response to get the list of teams"""

    teams: List[Union[TeamListItem, LiteLLM_TeamTable, LiteLLM_DeletedTeamTable]]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkTeamMemberAddRequest(BaseModel):
    """Request for bulk team member addition"""

    team_id: str
    members: Optional[List[Member]] = None  # List of members to add
    all_users: Optional[bool] = False  # Flag to add all users on Proxy to the team
    max_budget_in_team: Optional[float] = None


class TeamMemberAddResult(BaseModel):
    """Result of a single team member add operation"""

    user_id: Optional[str] = None
    user_email: Optional[str] = None
    success: bool
    error: Optional[str] = None
    updated_user: Optional[Dict[str, Any]] = None
    updated_team_membership: Optional[Dict[str, Any]] = None


class BulkTeamMemberAddResponse(BaseModel):
    """Response for bulk team member add operations"""

    team_id: str
    results: List[TeamMemberAddResult]
    total_requested: int
    successful_additions: int
    failed_additions: int
    updated_team: Optional[Dict[str, Any]] = None


class TeamMemberInfoResponse(LiteLLM_TeamMembership):
    """Response for GET /team/{team_id}/members/me — caller's own membership row."""

    role: Optional[str] = None
    user_email: Optional[str] = None
    team_alias: Optional[str] = None
