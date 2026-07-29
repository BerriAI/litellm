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
    model_itpm_limit: dict | None = None
    model_otpm_limit: dict | None = None
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

    @property
    def merged_metadata(self) -> dict:
        """Return metadata dict with dedicated rate-limit columns merged in.

        The ``metadata`` JSONB column holds ad-hoc key/value pairs, while
        ``model_itpm_limit``, ``model_otpm_limit``, ``model_rpm_limit``, and
        ``model_tpm_limit`` are dedicated first-class columns. Callers that read
        rate limits via ``project_metadata.get("model_itpm_limit")`` would
        silently get ``None`` if we only exposed the raw ``metadata`` field, so
        we merge the dedicated columns in here. Only non-None column values
        are included, so a column left at its default None does not shadow any
        legacy value stored in the ``metadata`` JSON blob.
        """
        base: dict = self.metadata or {}
        dedicated = {
            k: v
            for k, v in {
                "model_itpm_limit": self.model_itpm_limit,
                "model_otpm_limit": self.model_otpm_limit,
                "model_rpm_limit": self.model_rpm_limit,
                "model_tpm_limit": self.model_tpm_limit,
            }.items()
            if v is not None
        }
        return {**base, **dedicated}
