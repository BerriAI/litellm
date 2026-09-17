"""
End-user table model.

Canonical definition for ``litellm_endusertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from typing import Final, Literal

from pydantic import ConfigDict, model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_EndUserTable(LiteLLMPydanticObjectBase):
    user_id: str
    blocked: bool
    alias: str | None = None
    spend: float = 0.0
    allowed_model_region: Literal["eu", "us"] | None = None
    default_model: str | None = None
    budget_id: str | None = None
    litellm_budget_table: LiteLLM_BudgetTable | None = None
    object_permission_id: str | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values: object) -> object:
        if isinstance(values, cls):
            return values
        if isinstance(values, dict):
            if values.get("spend") is None:
                return {**values, "spend": 0.0}
            return values
        if hasattr(values, "model_dump") and callable(values.model_dump):
            data: Final = values.model_dump()
            if data.get("spend") is None:
                return {**data, "spend": 0.0}
            return data
        if hasattr(values, "__dict__"):
            data: Final = dict(values.__dict__)
            if data.get("spend") is None:
                return {**data, "spend": 0.0}
            return data
        return values

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
