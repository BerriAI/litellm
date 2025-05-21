from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr

from litellm.proxy._types import LiteLLM_UserTableWithKeyCount


class UserListResponse(BaseModel):
    """
    Response model for the user list endpoint
    """

    users: List[LiteLLM_UserTableWithKeyCount]
    total: int
    page: int
    page_size: int
    total_pages: int
