"""
Tag table model.

Canonical definition for ``litellm_tagtable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_TagTable(LiteLLMPydanticObjectBase):
    tag_name: str
    description: Optional[str] = None
    models: List[str] = []
    model_info: Optional[dict] = None
    spend: float = 0.0
    budget_id: Optional[str] = None
    litellm_budget_table: Optional[LiteLLM_BudgetTable] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("spend") is None:
            values.update({"spend": 0.0})
        if values.get("models") is None:
            values.update({"models": []})
        return values
