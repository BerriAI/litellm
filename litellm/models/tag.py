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
            raw = values.model_dump()  # rebind-ok: normalize model input
        elif hasattr(values, "__dict__") and not isinstance(values, dict):
            raw = dict(values.__dict__)  # rebind-ok: normalize object input  # mutable-ok: object normalization
        else:
            raw = values  # rebind-ok: preserve input

        if isinstance(raw, dict):
            if raw.get("spend") is None:
                raw.update({"spend": 0.0})  # mutable-ok: default spend value
            if raw.get("models") is None:
                raw.update({"models": []})  # mutable-ok: default models value
        return raw
