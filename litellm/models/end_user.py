"""
End-user table model.

Canonical definition for ``litellm_endusertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

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
        if isinstance(values, BaseModel):
            raw: Final = values.model_dump()  # mutable-ok: pydantic before-validator dict payload
        elif hasattr(values, "__dict__") and not isinstance(values, dict):
            raw: Final = dict(values.__dict__)  # mutable-ok: object normalization for ORM/Prisma models
        else:
            raw: Final = values

        if isinstance(raw, dict):
            if raw.get("spend") is None:
                return {**raw, "spend": 0.0}  # mutable-ok: default spend for uninitialized rows
            return raw
        return values

    model_config = ConfigDict(protected_namespaces=())
