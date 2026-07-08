from typing import List, Optional

from pydantic import BaseModel, Field

from litellm.models.budget import LiteLLM_BudgetTableFull
from litellm.models.end_user import LiteLLM_EndUserTable


class CustomerResponse(LiteLLM_EndUserTable):
    """Customer object returned by the /customer read+write endpoints.

    Nests the full budget response model so server-managed budget fields
    (budget_reset_at, created_at) survive response_model filtering, rather than
    the narrow write-allowlist shape LiteLLM_EndUserTable carries for internal use.
    """

    litellm_budget_table: Optional[LiteLLM_BudgetTableFull] = None  # pyright: ignore


class BlockUsersResponse(BaseModel):
    blocked_users: List[LiteLLM_EndUserTable]


class UnblockUsersResponse(BaseModel):
    blocked_users: List[str] = Field(description="User IDs that remain blocked after this unblock call")


class DeleteCustomersResponse(BaseModel):
    deleted_customers: int
    message: str
