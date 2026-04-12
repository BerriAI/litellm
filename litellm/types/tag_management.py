from typing import Dict, List, Optional

from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    description: Optional[str] = None
    models: Optional[List[str]] = None
    model_info: Optional[Dict[str, str]] = None  # maps model_id to model_name


class TagConfig(TagBase):
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class TagNewRequest(TagBase):
    budget_id: Optional[str] = None
    # Budget fields - if budget_id is None, create a new budget with these params
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_max_budget: Optional[Dict] = None
    budget_duration: Optional[str] = None


class TagUpdateRequest(TagBase):
    budget_id: Optional[str] = None
    # Budget fields - if provided, will update the budget
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_max_budget: Optional[Dict] = None
    budget_duration: Optional[str] = None


class TagDeleteRequest(BaseModel):
    name: str


class TagInfoRequest(BaseModel):
    names: List[str]
