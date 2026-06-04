"""
VerificationToken domain model.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from litellm.backend.models.base import DomainModel
from litellm.backend.models.object_permission import ObjectPermission


class VerificationToken(DomainModel):
    """Domain model for API verification tokens (virtual keys)."""

    token: Optional[str] = None
    key_name: Optional[str] = None
    key_alias: Optional[str] = None
    spend: float = 0.0
    max_budget: Optional[float] = None
    expires: Optional[datetime] = None
    models: List[str] = Field(default_factory=list)
    aliases: Dict[str, str] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_cache_controls: List[str] = Field(default_factory=list)
    allowed_routes: List[str] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    model_spend: Dict[str, float] = Field(default_factory=dict)
    model_max_budget: Dict[str, float] = Field(default_factory=dict)
    soft_budget_cooldown: bool = False
    blocked: Optional[bool] = None
    litellm_budget_table: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = None
    organization_id: Optional[str] = None
    budget_id: Optional[str] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    last_active: Optional[datetime] = None
    object_permission_id: Optional[str] = None
    object_permission: Optional[ObjectPermission] = None
    access_group_ids: List[str] = Field(default_factory=list)
    rotation_count: int = 0
    auto_rotate: bool = False
    rotation_interval: Optional[str] = None
    last_rotation_at: Optional[datetime] = None
    key_rotation_at: Optional[datetime] = None
    router_settings: Optional[Dict[str, Any]] = None
    budget_limits: Optional[List[Dict[str, Any]]] = None

    @field_validator("expires", mode="before")
    @classmethod
    def parse_expires(cls, v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @property
    def is_blocked(self) -> bool:
        return self.blocked is True

    @property
    def is_expired(self) -> bool:
        if self.expires is None:
            return False
        return datetime.utcnow() > self.expires

    def is_over_budget(self) -> bool:
        """Check if token spend exceeds max budget."""
        if self.max_budget is None:
            return False
        return self.spend >= self.max_budget

    def has_model_access(self, model_name: str) -> bool:
        """Check if the token has access to a specific model."""
        if not self.models:
            return True
        return model_name in self.models

    def has_route_access(self, route: str) -> bool:
        """Check if the token has access to a specific route."""
        if not self.allowed_routes:
            return True
        return route in self.allowed_routes


class DeletedVerificationToken(VerificationToken):
    """Audit record for deleted verification tokens."""

    id: Optional[str] = None
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    deleted_by_api_key: Optional[str] = None
    litellm_changed_by: Optional[str] = None
