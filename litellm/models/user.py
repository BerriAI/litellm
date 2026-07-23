"""
User table model.

Canonical definition for ``litellm_usertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.organization_membership import (
    LiteLLM_OrganizationMembershipTable,
)
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_UserTable(LiteLLMPydanticObjectBase):
    user_id: str
    user_alias: Optional[str] = None
    team_id: Optional[str] = None
    sso_user_id: Optional[str] = None
    organization_id: Optional[str] = None
    object_permission_id: Optional[str] = None
    password: Optional[str] = Field(default=None, exclude=True)
    teams: List[str] = []
    user_role: Optional[str] = None
    max_budget: Optional[float] = None
    spend: float = 0.0
    user_email: Optional[str] = None
    models: list = []
    metadata: Optional[dict] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_cache_controls: List[str] = []
    policies: List[str] = []
    model_spend: Optional[Dict] = {}
    model_max_budget: Optional[Dict] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    password_updated_at: Optional[datetime] = None
    reset_password_required: bool = False
    organization_memberships: Optional[List[LiteLLM_OrganizationMembershipTable]] = None
    object_permission: Optional[LiteLLM_ObjectPermissionTable] = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("spend") is None:
            values.update({"spend": 0.0})
        if values.get("models") is None:
            values.update({"models": []})
        if values.get("teams") is None:
            values.update({"teams": []})
        return values

    def is_over_budget(self) -> bool:
        if self.max_budget is None:
            return False
        return self.spend >= self.max_budget

    def has_model_access(self, model_name: str) -> bool:
        if not self.models:
            return True
        return model_name in self.models
