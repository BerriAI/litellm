"""
Organization membership table model.

Canonical definition for ``litellm_organizationmembership``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import ConfigDict, model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_OrganizationMembershipTable(LiteLLMPydanticObjectBase):
    """Tracks which organizations a user belongs to and their spend within it."""

    user_id: str
    organization_id: str
    user_role: Optional[str] = None
    spend: float = 0.0
    budget_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    user: Optional[Any] = None
    litellm_budget_table: Optional[LiteLLM_BudgetTable] = None
    user_email: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def populate_user_email(self) -> "LiteLLM_OrganizationMembershipTable":
        if self.user_email is None and self.user is not None:
            if isinstance(self.user, dict):
                self.user_email = self.user.get("user_email")
            else:
                self.user_email = getattr(self.user, "user_email", None)
        return self
