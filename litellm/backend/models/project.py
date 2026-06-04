"""
Project domain model.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from litellm.backend.models.base import DomainModel


class Project(DomainModel):
    """Domain model for projects."""

    project_id: Optional[str] = None
    project_alias: Optional[str] = None
    description: Optional[str] = None
    team_id: Optional[str] = None
    budget_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    models: List[str] = Field(default_factory=list)
    spend: float = 0.0
    model_spend: Dict[str, float] = Field(default_factory=dict)
    model_rpm_limit: Dict[str, int] = Field(default_factory=dict)
    model_tpm_limit: Dict[str, int] = Field(default_factory=dict)
    blocked: bool = False
    object_permission_id: Optional[str] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked
