"""
User domain model.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from litellm.backend.models.base import DomainModel


class User(DomainModel):
    """Domain model for users."""

    user_id: str
    user_alias: Optional[str] = None
    team_id: Optional[str] = None
    sso_user_id: Optional[str] = None
    organization_id: Optional[str] = None
    object_permission_id: Optional[str] = None
    password: Optional[str] = None
    teams: List[str] = Field(default_factory=list)
    user_role: Optional[str] = None
    max_budget: Optional[float] = None
    spend: float = 0.0
    user_email: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_cache_controls: List[str] = Field(default_factory=list)
    policies: List[str] = Field(default_factory=list)
    model_spend: Dict[str, float] = Field(default_factory=dict)
    model_max_budget: Dict[str, float] = Field(default_factory=dict)

    def is_over_budget(self) -> bool:
        """Check if user spend exceeds max budget."""
        if self.max_budget is None:
            return False
        return self.spend >= self.max_budget

    def has_model_access(self, model_name: str) -> bool:
        """Check if the user has access to a specific model."""
        if not self.models:
            return True
        return model_name in self.models
