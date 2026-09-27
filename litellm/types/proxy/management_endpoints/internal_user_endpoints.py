from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.proxy._types import (
    LiteLLM_UserTableWithKeyCount,
    NewUserRequest,
    UpdateUserRequest,
    UpdateUserRequestNoUserIDorEmail,
)
from litellm.types.proxy.management_endpoints.management_v1 import ResourceResponse

MAX_BULK_DELETE_USERS: Final = 500

MAX_BULK_NEW_USERS: Final = 500


class InsensitiveContains(TypedDict):
    contains: ReadOnly[str]
    mode: ReadOnly[Literal["insensitive"]]


class UserSearchWhere(TypedDict):
    """Prisma filter behind `/user/list?search=`: user_id or user_email contains the term, case-insensitive."""

    OR: ReadOnly[tuple[Mapping[Literal["user_id", "user_email"], InsensitiveContains], ...]]


class KeyActivitySearchWhere(TypedDict):
    """Prisma filter behind `/user/daily/activity/aggregated/search`: exact token hash, or key alias
    or user id containing the term, case-insensitive."""

    user_id: NotRequired[ReadOnly[str]]
    OR: ReadOnly[
        tuple[Mapping[Literal["token"], str] | Mapping[Literal["key_alias", "user_id"], InsensitiveContains], ...]
    ]


class UserListResponse(BaseModel):
    """
    Response model for the user list endpoint
    """

    users: list[LiteLLM_UserTableWithKeyCount]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkUpdateUserRequest(BaseModel):
    """Request for bulk user updates"""

    users: list[UpdateUserRequest] | None = None  # List of specific user update requests
    all_users: bool | None = False  # Flag to update all users
    user_updates: UpdateUserRequestNoUserIDorEmail | None = None  # Updates to apply to all users when all_users=True

    @field_validator("users", "all_users", "user_updates")
    @classmethod
    def validate_request(cls, v, info):
        # Get all field values for validation
        values: Final = info.data if hasattr(info, "data") else {}

        # After all fields are set, validate the combination
        if info.field_name == "user_updates":  # This is the last field, do validation here
            users: Final = values.get("users")
            all_users: Final = values.get("all_users", False)
            user_updates: Final = v

            # Must specify either users list OR all_users with user_updates
            if not users and not (all_users and user_updates):
                raise ValueError(
                    "Must specify either 'users' for individual updates or 'all_users=True' with 'user_updates' for bulk updates"
                )

            # Cannot specify both users list and all_users
            if users and all_users:
                raise ValueError("Cannot specify both 'users' and 'all_users=True'. Choose one approach.")

        return v


class UserUpdateResult(BaseModel):
    """Result of a single user update operation"""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None
    updated_user: dict[str, Any] | None = None


class BulkUpdateUserResponse(BaseModel):
    """Response for bulk user update operations"""

    results: list[UserUpdateResult]
    total_requested: int
    successful_updates: int
    failed_updates: int


class BulkDeleteUserRequest(BaseModel):
    """Body of `POST /management/v1/users/bulk_delete`."""

    model_config = ConfigDict(extra="forbid")

    user_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_BULK_DELETE_USERS)


class UserDeleteResult(BaseModel):
    """Outcome for one requested user, in request order. `teams_removed` lists the teams the user left."""

    user_id: str
    user_email: str | None = None
    success: bool
    teams_removed: tuple[str, ...] = ()
    error: str | None = None


class BulkDeleteUsersResponse(ResourceResponse[tuple[UserDeleteResult, ...]]):
    """`{data: [...]}` with one `UserDeleteResult` per requested user, in request order."""


class BulkNewUserItem(NewUserRequest):
    """One row of `POST /management/v1/users/bulk`: the `/user/new` body, with keys opt-in and invite emails
    unsupported. Unknown fields are rejected, as on every `/management/v1` request body."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    auto_create_key: bool = False

    @field_validator("send_invite_email")
    @classmethod
    def reject_invite_email(cls, value: bool | None) -> bool | None:
        if value:
            raise ValueError("send_invite_email is not supported on /management/v1/users/bulk; invite users separately")
        return value


class BulkNewUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    users: Sequence[BulkNewUserItem] = Field(min_length=1, max_length=MAX_BULK_NEW_USERS)


class UserCreateResult(BaseModel):
    """Outcome for one row of `POST /management/v1/users/bulk`. `teams` lists the teams the user was actually
    added to."""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    teams: tuple[str, ...] | None = None
    key: str | None = None
    error: str | None = None


class BulkNewUserMeta(BaseModel):
    total_requested: int
    created: int
    failed: int


class BulkNewUserResponse(BaseModel):
    """`data` holds one result per input row, in input order."""

    data: tuple[UserCreateResult, ...]
    meta: BulkNewUserMeta


UserDailyActivityExportType = Literal["daily", "daily_with_keys", "daily_with_models"]


class UserDailyActivityExportRow(BaseModel):
    date: str
    user_id: str
    user_email: str | None = None
    user_alias: str | None = None
    api_key: str | None = None
    key_alias: str | None = None
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


class UserDailyActivityExportMetadata(BaseModel):
    export_date: str
    export_type: UserDailyActivityExportType
    start_date: str
    end_date: str
    user_id: str | None
    total_spend: float
    total_flat_cost: float = 0.0
    total_api_requests: int
    total_successful_requests: int
    total_failed_requests: int
    total_tokens: int


class UserDailyActivityExportResponse(BaseModel):
    metadata: UserDailyActivityExportMetadata
    data: list[UserDailyActivityExportRow]
