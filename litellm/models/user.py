"""
User table model.

Canonical definition for ``litellm_usertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime

from pydantic import ConfigDict, Field, model_validator

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.organization_membership import (
    LiteLLM_OrganizationMembershipTable,
)
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_UserTable(LiteLLMPydanticObjectBase):
    user_id: str
    user_alias: str | None = None
    team_id: str | None = None
    sso_user_id: str | None = None
    organization_id: str | None = None
    object_permission_id: str | None = None
    password: str | None = Field(default=None, exclude=True)
    teams: list[str] = []
    user_role: str | None = None
    max_budget: float | None = None
    spend: float = 0.0
    user_email: str | None = None
    models: list = []
    metadata: dict | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    allowed_cache_controls: list[str] = []
    policies: list[str] = []
    model_spend: dict | None = {}
    model_max_budget: dict | None = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None
    organization_memberships: list[LiteLLM_OrganizationMembershipTable] | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None

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
