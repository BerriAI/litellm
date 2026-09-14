from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import (
    LiteLLM_UserTableWithKeyCount,
    NewUserRequest,
    UpdateUserRequest,
    UpdateUserRequestNoUserIDorEmail,
)

MAX_BULK_NEW_USERS: Final = 500


class InsensitiveContains(TypedDict):
    contains: ReadOnly[str]
    mode: ReadOnly[Literal["insensitive"]]


class UserSearchWhere(TypedDict):
    """Prisma filter behind `/user/list?search=`: user_id or user_email contains the term, case-insensitive."""

    OR: ReadOnly[tuple[Mapping[Literal["user_id", "user_email"], InsensitiveContains], ...]]


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


class BulkNewUserItem(NewUserRequest):
    """One row of `/user/bulk_new`: the `/user/new` body, with keys opt-in and invite emails unsupported."""

    auto_create_key: bool = False

    @field_validator("send_invite_email")
    @classmethod
    def reject_invite_email(cls, value: bool | None) -> bool | None:
        if value:
            raise ValueError("send_invite_email is not supported on /user/bulk_new; invite users separately")
        return value


class BulkNewUserRequest(BaseModel):
    users: tuple[BulkNewUserItem, ...] = Field(min_length=1, max_length=MAX_BULK_NEW_USERS)


class UserCreateResult(BaseModel):
    """Outcome for one row of `/user/bulk_new`. `teams` lists the teams the user was actually added to."""

    user_id: str | None = None
    user_email: str | None = None
    success: bool
    teams: tuple[str, ...] | None = None
    key: str | None = None
    error: str | None = None


class BulkNewUserResponse(BaseModel):
    results: tuple[UserCreateResult, ...]
    total_requested: int
    successful_creations: int
    failed_creations: int
