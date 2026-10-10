from litellm.proxy._types import BudgetNewRequest
from litellm.types.llms.base import LiteLLMBaseModel


class TagBase(LiteLLMBaseModel):
    name: str
    description: str | None = None
    models: list[str] | None = None
    model_info: dict[str, str] | None = None  # maps model_id to model_name


class TagConfig(TagBase):
    created_at: str
    updated_at: str
    created_by: str | None = None


class TagListItem(LiteLLMBaseModel):
    """One entry of the GET /tag/list response: a stored tag, or a dynamic tag seen only in spend."""

    name: str
    description: str | None = None
    # None for dynamic tags, which control no models
    models: list[str] | None = None
    # stored as free-form JSON, so values are not guaranteed to be strings
    model_info: dict[str, object] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    created_by: str | None = None
    litellm_budget_table: BudgetNewRequest | None = None


class TagNewRequest(TagBase):
    budget_id: str | None = None
    # Budget fields - if budget_id is None, create a new budget with these params
    max_budget: float | None = None
    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    model_max_budget: dict | None = None
    budget_duration: str | None = None


class TagUpdateRequest(TagBase):
    budget_id: str | None = None
    # Budget fields - if provided, will update the budget
    max_budget: float | None = None
    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    model_max_budget: dict | None = None
    budget_duration: str | None = None


class TagDeleteRequest(LiteLLMBaseModel):
    name: str


class TagInfoRequest(LiteLLMBaseModel):
    names: list[str]
