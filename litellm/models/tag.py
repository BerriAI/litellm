"""
Tag table model.

Canonical definition for ``litellm_tagtable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Final

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
    def set_model_info(cls, values: object) -> object:
        if isinstance(values, BaseModel):
            raw: Final = values.model_dump()  # mutable-ok: pydantic before-validator dict payload
        elif hasattr(values, "__dict__") and not isinstance(values, dict):
            raw: Final = dict(values.__dict__)  # mutable-ok: object normalization for ORM/Prisma models
        else:
            raw: Final = values

        if isinstance(raw, dict):
            updates: Final = {  # mutable-ok: default values for uninitialized fields
                **({"spend": 0.0} if raw.get("spend") is None else {}),  # mutable-ok: conditional default
                **({"models": []} if raw.get("models") is None else {}),  # mutable-ok: conditional default
            }
            if updates:
                return {**raw, **updates}  # mutable-ok: merged normalized model dictionary
            return raw
        return values
