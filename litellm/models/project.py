"""
Project table model.

Canonical definition for ``litellm_projecttable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import List, Optional

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_ProjectTable(LiteLLMPydanticObjectBase):
    """Database model representation for project"""

    project_id: str
    project_alias: Optional[str] = None
    description: Optional[str] = None
    team_id: Optional[str] = None
    budget_id: Optional[str] = None
    metadata: Optional[dict] = None
    models: List[str] = []
    spend: float = 0.0
    model_spend: Optional[dict] = None
    model_rpm_limit: Optional[dict] = None
    model_tpm_limit: Optional[dict] = None
    blocked: bool = False
    object_permission_id: Optional[str] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    litellm_budget_table: Optional[LiteLLM_BudgetTable] = None
    object_permission: Optional[LiteLLM_ObjectPermissionTable] = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked
