"""
Organization table model.

Canonical definition for ``litellm_organizationtable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import json
from typing import Final

from pydantic import BaseModel, model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase

_JSON_OBJECT_COLUMNS: Final = frozenset({"metadata", "model_spend"})


def _decode_json_object_column(field: str, value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Field {field} should be a valid dictionary") from e


class LiteLLM_OrganizationTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_OrganizationTable record"""

    organization_id: str | None = None
    organization_alias: str | None = None
    budget_id: str
    spend: float = 0.0
    metadata: dict | None = None
    models: list[str] = []
    model_spend: dict | None = {}
    created_by: str
    updated_by: str
    users: list[LiteLLM_UserTable] | None = None
    litellm_budget_table: LiteLLM_BudgetTable | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None
    object_permission_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def decode_json_object_columns(cls, values: object) -> object:
        payload: Final = values.model_dump() if isinstance(values, BaseModel) else values
        if not isinstance(payload, dict):
            return values
        return {
            key: _decode_json_object_column(key, value) if key in _JSON_OBJECT_COLUMNS else value
            for key, value in payload.items()
        }
