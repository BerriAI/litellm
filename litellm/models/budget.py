"""
Budget domain model.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from litellm.models.base import DomainModel


class Budget(DomainModel):
    """Domain model for budget configuration."""

    budget_id: Optional[str] = None
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_max_budget: Optional[Dict[str, Any]] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_models: List[str] = Field(default_factory=list)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    def is_over_budget(self, current_spend: float) -> bool:
        """Check if current spend exceeds the max budget."""
        if self.max_budget is None:
            return False
        return current_spend >= self.max_budget

    def is_approaching_soft_budget(self, current_spend: float) -> bool:
        """Check if current spend is approaching soft budget threshold."""
        if self.soft_budget is None:
            return False
        return current_spend >= self.soft_budget

    def should_reset_budget(self) -> bool:
        """Check if budget should be reset based on budget_reset_at."""
        if self.budget_reset_at is None:
            return False
        return datetime.utcnow() >= self.budget_reset_at
