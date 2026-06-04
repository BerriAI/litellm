"""
Organization domain model.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from litellm.models.base import DomainModel


class Organization(DomainModel):
    """Domain model for organizations."""

    organization_id: Optional[str] = None
    organization_alias: str
    budget_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    models: List[str] = Field(default_factory=list)
    spend: float = 0.0
    model_spend: Dict[str, float] = Field(default_factory=dict)
    object_permission_id: Optional[str] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
