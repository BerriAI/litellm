"""
Organization table model.

Canonical definition for ``litellm_organizationtable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from typing import List, Optional

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_OrganizationTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_OrganizationTable record"""

    organization_id: Optional[str] = None
    organization_alias: Optional[str] = None
    budget_id: str
    spend: float = 0.0
    metadata: Optional[dict] = None
    models: List[str] = []
    model_spend: Optional[dict] = {}
    created_by: str
    updated_by: str
    users: Optional[List[LiteLLM_UserTable]] = None
    litellm_budget_table: Optional[LiteLLM_BudgetTable] = None
    object_permission: Optional[LiteLLM_ObjectPermissionTable] = None
    object_permission_id: Optional[str] = None
