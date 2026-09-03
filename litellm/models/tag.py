"""
Tag table model.

Canonical definition for ``litellm_tagtable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime

from pydantic import BaseModel, model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_TagTable(LiteLLMPydanticObjectBase):
    tag_name: str
    description: str | None = None
    models: list[str] = []
    model_info: dict | None = None
    spend: float = 0.0
    budget_id: str | None = None
    litellm_budget_table: LiteLLM_BudgetTable | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if isinstance(values, BaseModel):
            normalized = values.model_dump()
        elif hasattr(values, "__dict__") and not isinstance(values, dict):
            normalized = dict(values.__dict__)
        elif isinstance(values, dict):
            normalized = dict(values)
        else:
            return values

        if normalized.get("spend") is None:
            normalized["spend"] = 0.0
        if normalized.get("models") is None:
            normalized["models"] = []
        return normalized
