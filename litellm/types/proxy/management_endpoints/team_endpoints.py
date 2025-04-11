from typing import List, Optional

from pydantic import BaseModel


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
