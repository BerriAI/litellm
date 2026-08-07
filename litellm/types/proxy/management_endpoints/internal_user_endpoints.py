from typing import Any, Final

from pydantic import BaseModel, field_validator

from litellm.proxy._types import (
    LiteLLM_UserTableWithKeyCount,
    UpdateUserRequest,
    UpdateUserRequestNoUserIDorEmail,
)


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
