from pydantic import ConfigDict, Field

from litellm.models.budget import LiteLLM_BudgetTableFull
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.types.llms.base import LiteLLMBaseModel


class CustomerResponse(LiteLLM_EndUserTable):
    """Customer object returned by the /customer read+write endpoints.

    Nests the full budget response model so server-managed budget fields
    (budget_reset_at, created_at) survive response_model filtering, rather than
    the narrow write-allowlist shape LiteLLM_EndUserTable carries for internal use.
    """

    litellm_budget_table: LiteLLM_BudgetTableFull | None = None  # pyright: ignore


class BlockUsersResponse(LiteLLMBaseModel):
    blocked_users: list[LiteLLM_EndUserTable]


class UnblockUsersResponse(LiteLLMBaseModel):
    blocked_users: list[str] = Field(description="User IDs that remain blocked after this unblock call")


class DeleteCustomersResponse(LiteLLMBaseModel):
    deleted_customers: int
    message: str


class CustomerUpdateResult(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    success: bool
    updated_customer: CustomerResponse | None = None
    error: str | None = None


class BulkUpdateCustomerResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[CustomerUpdateResult, ...]
    total_requested: int
    successful_updates: int
    failed_updates: int
