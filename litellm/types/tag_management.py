from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    description: str | None = None
    models: list[str] | None = None
    model_info: dict[str, str] | None = None  # maps model_id to model_name


class TagConfig(TagBase):
    created_at: str
    updated_at: str
    created_by: str | None = None


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


class TagDeleteRequest(BaseModel):
    name: str


class TagInfoRequest(BaseModel):
    names: list[str]
