from typing import List

from pydantic import BaseModel

from litellm.models.end_user import LiteLLM_EndUserTable


class BlockUsersResponse(BaseModel):
    blocked_users: List[LiteLLM_EndUserTable]


class UnblockUsersResponse(BaseModel):
    blocked_users: List[str]


class DeleteCustomersResponse(BaseModel):
    deleted_customers: int
    message: str
