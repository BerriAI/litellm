from typing import List

from pydantic import BaseModel, Field

from litellm.models.end_user import LiteLLM_EndUserTable


class BlockUsersResponse(BaseModel):
    blocked_users: List[LiteLLM_EndUserTable]


class UnblockUsersResponse(BaseModel):
    blocked_users: List[str] = Field(
        description="User IDs that remain blocked after this unblock call"
    )


class DeleteCustomersResponse(BaseModel):
    deleted_customers: int
    message: str
