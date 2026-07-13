"""
PTU Reservation table model.

Canonical definition for ``litellm_ptureservation``.
"""

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, model_validator

from litellm.types.llms.base import LiteLLMPydanticObjectBase

CostSource = Literal["manual", "azure_billing"]


class LiteLLM_PTUReservation(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_PTUReservation record."""

    id: str | None = None
    team_id: str
    model: str
    cost_source: CostSource = "manual"

    ptu_count: int | None = None
    cost_per_ptu: float | None = None

    azure_resource_id: str | None = None

    effective_from: datetime
    effective_to: datetime | None = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def _enforce_cost_source_fields(self) -> "LiteLLM_PTUReservation":
        if self.cost_source == "manual":
            if self.ptu_count is None or self.cost_per_ptu is None:
                raise ValueError("manual reservations require both ptu_count and cost_per_ptu")
            if self.ptu_count <= 0:
                raise ValueError("ptu_count must be positive")
            if self.cost_per_ptu < 0:
                raise ValueError("cost_per_ptu must be non-negative")
        elif self.cost_source == "azure_billing":
            if self.azure_resource_id is None:
                raise ValueError("azure_billing reservations require azure_resource_id")
            if self.ptu_count is not None or self.cost_per_ptu is not None:
                raise ValueError("azure_billing reservations must not set ptu_count or cost_per_ptu")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be strictly after effective_from")
        return self


class LiteLLM_PTUReservationFull(LiteLLM_PTUReservation):
    """LiteLLM_PTUReservation + server-managed fields returned on API responses."""

    created_by: str
    created_at: datetime
    updated_by: str
    updated_at: datetime
