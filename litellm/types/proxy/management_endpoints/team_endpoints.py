from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from litellm.proxy._types import (
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


class TeamListResponse(BaseModel):
    """Response to get the list of teams"""

    teams: List[LiteLLM_TeamTable]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkTeamMemberAddRequest(BaseModel):
    """Request for bulk team member addition"""

    team_id: str
    members: List[Member]  # List of members to add
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
