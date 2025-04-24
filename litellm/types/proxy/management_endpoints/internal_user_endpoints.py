from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr

if TYPE_CHECKING:
    from litellm_proxy._types import LiteLLM_UserTableWithKeyCount
else:
    LiteLLM_UserTableWithKeyCount = Any


class UserListResponse(BaseModel):
    """
    Response model for the user list endpoint
    """

    users: List[LiteLLM_UserTableWithKeyCount]
    total: int
    page: int
    page_size: int
    total_pages: int
