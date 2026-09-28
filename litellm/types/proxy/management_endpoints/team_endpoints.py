from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.proxy._types import (
    KeyManagementRoutes,
    LiteLLM_DeletedTeamTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    Member,
    MemberDeleteRequest,
)
from litellm.proxy.common_utils.timezone_utils import budget_duration_error
from litellm.types.proxy.management_endpoints.internal_user_endpoints import InsensitiveContains
from litellm.types.proxy.management_endpoints.management_v1 import ResourceResponse

TeamIdSearchMatch = Literal["exact", "prefix"]


TeamIdSearchFilter = TypedDict(
    "TeamIdSearchFilter",
    {  # mutable-ok: functional TypedDict field map
        "in": NotRequired[ReadOnly[Sequence[str]]],
        "notIn": NotRequired[ReadOnly[Sequence[str]]],
    },
)


class TeamKeyActivitySearchWhere(TypedDict):
    """Prisma filter behind `/team/daily/activity/aggregated/search`: exact token hash, or key alias
    or user id containing the term, case-insensitive, narrowed to the teams and keys the caller may see."""

    team_id: NotRequired[ReadOnly[TeamIdSearchFilter]]
    token: NotRequired[ReadOnly[Mapping[Literal["in"], Sequence[str]]]]
    OR: ReadOnly[
        tuple[Mapping[Literal["token"], str] | Mapping[Literal["key_alias", "user_id"], InsensitiveContains], ...]
    ]


MAX_BULK_TEAM_MEMBER_DELETES: Final = 500

MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES: Final = 500


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


class TeamMemberRef(MemberDeleteRequest):
    """One member, named by exactly one of `user_id` or `user_email`."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def one_identifier(self) -> "TeamMemberRef":
        if self.user_id is not None and self.user_email is not None:
            raise ValueError("Each member must be identified by exactly one of user_id or user_email")
        return self


class BulkTeamMemberDeleteRequest(BaseModel):
    """Body of `POST /management/v1/teams/{team_id}/members/bulk_delete`."""

    model_config = ConfigDict(extra="forbid")

    members: tuple[TeamMemberRef, ...] = Field(min_length=1, max_length=MAX_BULK_TEAM_MEMBER_DELETES)


class TeamMemberDeleteResult(BaseModel):
    """Outcome for one requested member, in request order."""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None


class BulkTeamMemberDeleteResponse(ResourceResponse[tuple[TeamMemberDeleteResult, ...]]):
    """`{data: [...]}` with one `TeamMemberDeleteResult` per requested member, in request order."""


class TeamMemberBudgetPatch(TeamMemberRef):
    """One member's per-member limits, merge-patch style: a field left out of the row is
    untouched, a field sent as null is cleared, and clearing the last limit drops the
    member back to the team default."""

    max_budget_in_team: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: tuple[str, ...] | None = None

    @field_validator("budget_duration")
    @classmethod
    def persistable_budget_duration(cls, value: str | None) -> str | None:
        error: Final = budget_duration_error(value)
        if error is not None:
            raise ValueError(error)
        return value


class BulkTeamMemberBudgetUpdateRequest(BaseModel):
    """Body of `POST /management/v1/teams/{team_id}/members/bulk_update`."""

    model_config = ConfigDict(extra="forbid")

    members: tuple[TeamMemberBudgetPatch, ...] = Field(min_length=1, max_length=MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES)


class TeamMemberBudgetUpdateResult(BaseModel):
    """Outcome for one requested member, in request order, carrying the limits in force
    after the write rather than the ones that were asked for."""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None
    budget_id: str | None = None
    max_budget: float | None = None
    max_budget_source: Literal["member", "team_default"] | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: tuple[str, ...] | None = None


class BulkTeamMemberBudgetUpdateResponse(ResourceResponse[tuple[TeamMemberBudgetUpdateResult, ...]]):
    """`{data: [...]}` with one `TeamMemberBudgetUpdateResult` per requested member, in request order."""


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


class TeamUserSpendRow(BaseModel):
    team_id: str
    team_alias: str | None = None
    user_id: str
    user_email: str | None = None
    user_alias: str | None = None
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class TeamUserSpendResponse(BaseModel):
    start_date: str
    end_date: str
    results: tuple[TeamUserSpendRow, ...]


TeamDailyActivityExportType = Literal["daily", "daily_with_keys", "daily_with_users", "daily_with_models"]
TeamDailyActivityExportFormat = Literal["csv", "json"]


class TeamDailyActivityExportRow(BaseModel):
    date: str
    team_id: str
    team_alias: str | None = None
    api_key: str | None = None
    key_alias: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    keys: int | None = None
    model: str | None = None
    spend: float
    flat_cost: float = 0.0
    api_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class TeamDailyActivityExportMetadata(BaseModel):
    export_date: str
    export_type: TeamDailyActivityExportType
    start_date: str
    end_date: str
    team_ids: list[str] | None
    total_spend: float
    total_flat_cost: float = 0.0
    total_api_requests: int
    total_successful_requests: int
    total_failed_requests: int
    total_tokens: int


class TeamDailyActivityExportResponse(BaseModel):
    metadata: TeamDailyActivityExportMetadata
    data: list[TeamDailyActivityExportRow]
