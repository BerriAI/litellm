"""
Project table model.

Canonical definition for ``litellm_projecttable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_ProjectTable(LiteLLMPydanticObjectBase):
    """Database model representation for project"""

    project_id: str
    project_alias: str | None = None
    description: str | None = None
    team_id: str | None = None
    budget_id: str | None = None
    metadata: dict | None = None
    models: list[str] = []
    spend: float = 0.0
    model_spend: dict | None = None
    model_rpm_limit: dict | None = None
    model_tpm_limit: dict | None = None
    blocked: bool = False
    object_permission_id: str | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    litellm_budget_table: LiteLLM_BudgetTable | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked
