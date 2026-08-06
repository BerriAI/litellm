from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy._types import (
    KeyManagementRoutes,
    LiteLLM_DeletedTeamTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    Member,
)

TeamIdSearchMatch = Literal["exact", "prefix"]


class GetTeamMemberPermissionsRequest(BaseModel):
    """Request to get the team member permissions for a team"""

    team_id: str


class GetTeamMemberPermissionsResponse(BaseModel):
    """Response to get the team member permissions for a team"""

    team_id: str
    """
    The team id that the permissions are for
    """

    team_member_permissions: list[str] | None = []
    """
    The team member permissions currently set for the team
    """

    all_available_permissions: list[str]
    """
    All available team member permissions
    """


class UpdateTeamMemberPermissionsRequest(BaseModel):
    """Request to update the team member permissions for a team"""

    team_id: str
    team_member_permissions: list[str]


class BulkUpdateTeamMemberPermissionsRequest(BaseModel):
    """Request to bulk-update team member permissions across teams."""

    permissions: list[KeyManagementRoutes]
    """Permissions to append to the target teams (duplicates are skipped)."""

    team_ids: list[str] | None = None
    """Specific team IDs to update. Required unless apply_to_all_teams is True."""

    apply_to_all_teams: bool = False
    """When True, update all teams. Mutually exclusive with team_ids."""


class BulkUpdateTeamMemberPermissionsResponse(BaseModel):
    """Response for bulk team member permissions update."""

    message: str
    teams_updated: int
    permissions_appended: list[str] | None = None


class TeamListItem(LiteLLM_TeamTable):
    """A team item in the paginated list response, enriched with computed fields."""

    members_count: int = 0
    keys_count: int = 0
    # Resources inherited from access groups (separate from direct assignments)
    access_group_models: list[str] | None = None
    access_group_mcp_server_ids: list[str] | None = None
    access_group_agent_ids: list[str] | None = None


class TeamListResponse(BaseModel):
    """Response to get the list of teams"""

    teams: list[TeamListItem | LiteLLM_TeamTable | LiteLLM_DeletedTeamTable]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkTeamMemberAddRequest(BaseModel):
    """Request for bulk team member addition"""

    team_id: str
    members: list[Member] | None = None  # List of members to add
    all_users: bool | None = False  # Flag to add all users on Proxy to the team
    max_budget_in_team: float | None = None


class TeamMemberAddResult(BaseModel):
    """Result of a single team member add operation"""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None
    updated_user: dict[str, Any] | None = None
    updated_team_membership: dict[str, Any] | None = None


class BulkTeamMemberAddResponse(BaseModel):
    """Response for bulk team member add operations"""

    team_id: str
    results: list[TeamMemberAddResult]
    total_requested: int
    successful_additions: int
    failed_additions: int
    updated_team: dict[str, Any] | None = None


class TeamMemberInfoResponse(LiteLLM_TeamMembership):
    """Response for GET /team/{team_id}/members/me — caller's own membership row."""

    role: str | None = None
    user_email: str | None = None
    team_alias: str | None = None


class TeamMetadataFieldSchema(BaseModel):
    """One declared team metadata field from ``general_settings.team_metadata_schema``.

    Advisory only: the UI uses it to prepopulate the team metadata form.
    Enforcement stays with ``custom_team_metadata_validate``.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    label: str | None = None


class TeamMetadataSchemaResponse(BaseModel):
    """Response for GET /team/metadata_schema; ``fields`` is empty when no schema is configured."""

    fields: tuple[TeamMetadataFieldSchema, ...]
