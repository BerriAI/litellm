from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr

from litellm.proxy._types import LiteLLM_UserTableWithKeyCount, UpdateUserRequest


class UserListResponse(BaseModel):
    """
    Response model for the user list endpoint
    """

    users: List[LiteLLM_UserTableWithKeyCount]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkUpdateUserRequest(BaseModel):
    """Request for bulk user updates"""

    users: List[UpdateUserRequest]  # List of user update requests


class UserUpdateResult(BaseModel):
    """Result of a single user update operation"""

    user_id: Optional[str] = None
    user_email: Optional[str] = None
    success: bool
    error: Optional[str] = None
    updated_user: Optional[Dict[str, Any]] = None


class BulkUpdateUserResponse(BaseModel):
    """Response for bulk user update operations"""

    results: List[UserUpdateResult]
    total_requested: int
    successful_updates: int
    failed_updates: int
