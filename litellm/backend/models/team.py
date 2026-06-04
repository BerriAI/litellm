"""
Team domain model.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from litellm.backend.models.base import DomainModel
from litellm.backend.models.object_permission import ObjectPermission


class TeamMember(DomainModel):
    """Representation of a team member with role."""

    user_id: str
    role: str = "user"


class Team(DomainModel):
    """Domain model for teams."""

    team_id: str
    team_alias: Optional[str] = None
    organization_id: Optional[str] = None
    object_permission_id: Optional[str] = None
    admins: List[str] = Field(default_factory=list)
    members: List[str] = Field(default_factory=list)
    members_with_roles: List[TeamMember] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    spend: float = 0.0
    models: List[str] = Field(default_factory=list)
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    blocked: bool = False
    model_spend: Dict[str, float] = Field(default_factory=dict)
    model_max_budget: Dict[str, float] = Field(default_factory=dict)
    router_settings: Optional[Dict[str, Any]] = None
    team_member_permissions: List[str] = Field(default_factory=list)
    access_group_ids: List[str] = Field(default_factory=list)
    policies: List[str] = Field(default_factory=list)
    default_team_member_models: List[str] = Field(default_factory=list)
    budget_limits: Optional[Dict[str, Any]] = None
    model_id: Optional[int] = None
    allow_team_guardrail_config: bool = False
    object_permission: Optional[ObjectPermission] = None

    @field_validator("members_with_roles", mode="before")
    @classmethod
    def parse_members_with_roles(cls, v: Any) -> List[TeamMember]:
        if v is None or v == {} or v == []:
            return []
        if isinstance(v, dict):
            return [
                (
                    TeamMember(user_id=uid, role=r)
                    if isinstance(r, str)
                    else TeamMember(user_id=uid, **r)
                )
                for uid, r in v.items()
            ]
        if isinstance(v, list):
            return [TeamMember(**m) if isinstance(m, dict) else m for m in v]
        return v

    @property
    def is_blocked(self) -> bool:
        return self.blocked

    def is_admin(self, user_id: str) -> bool:
        """Check if a user is an admin of this team."""
        return user_id in self.admins

    def is_member(self, user_id: str) -> bool:
        """Check if a user is a member of this team."""
        return user_id in self.members or user_id in self.admins

    def has_model_access(self, model_name: str) -> bool:
        """Check if the team has access to a specific model."""
        if not self.models:
            return True
        return model_name in self.models

    def is_over_budget(self) -> bool:
        """Check if team spend exceeds max budget."""
        if self.max_budget is None:
            return False
        return self.spend >= self.max_budget


class CachedTeam(Team):
    """Team model with caching metadata."""

    last_refreshed_at: Optional[float] = None


class DeletedTeam(Team):
    """Audit record for deleted teams."""

    id: Optional[str] = None
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    deleted_by_api_key: Optional[str] = None
    litellm_changed_by: Optional[str] = None
