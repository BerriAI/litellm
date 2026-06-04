"""
Model domain model for proxy models.
"""

from typing import Any, Dict, Optional

from litellm.models.base import DomainModel


class Model(DomainModel):
    """Domain model for LLM models registered on the proxy."""

    model_id: Optional[str] = None
    model_name: str
    litellm_params: Dict[str, Any]
    model_info: Optional[Dict[str, Any]] = None
    blocked: bool = False
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked

    @property
    def team_id(self) -> Optional[str]:
        """Extract team_id from model_info if present."""
        if self.model_info:
            return self.model_info.get("team_id")
        return None

    @property
    def team_public_model_name(self) -> Optional[str]:
        """Extract team_public_model_name from model_info if present."""
        if self.model_info:
            return self.model_info.get("team_public_model_name")
        return None
