"""Request/response types for /ptu_reservation/* endpoints."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from litellm.types.llms.base import LiteLLMPydanticObjectBase


def _to_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class PTUReservationNewRequest(LiteLLMPydanticObjectBase):
    """Body accepted by POST /ptu_reservation/new."""

    team_id: str = Field(description="Team that owns the reservation.")
    model: str = Field(description="Model name the reservation covers.")
    cost_source: Literal["manual", "azure_billing"] = Field(
        default="manual",
        description="Source of the cost figures. Only 'manual' is supported in this release.",
    )
    ptu_count: int | None = Field(
        default=None,
        description="Number of provisioned throughput units. Required for manual.",
    )
    cost_per_ptu: float | None = Field(
        default=None,
        description="Monthly cost per PTU in USD. Required for manual.",
    )
    azure_resource_id: str | None = Field(
        default=None,
        description="Azure resource ARM id, for cost_source='azure_billing'.",
    )
    effective_from: datetime = Field(description="Inclusive UTC start of the reservation.")
    effective_to: datetime | None = Field(
        default=None,
        description="Exclusive UTC end of the reservation. Null = still active.",
    )

    model_config = ConfigDict(protected_namespaces=())

    @field_validator("effective_from", "effective_to", mode="after")
    @classmethod
    def _coerce_utc(cls, value: datetime | None) -> datetime | None:
        return _to_utc(value)


class PTUReservationListRequest(LiteLLMPydanticObjectBase):
    """Query params for GET /ptu_reservation/list."""

    team_id: str | None = None
    model: str | None = None
    active_only: bool = False


class PTUReservationCloseRequest(LiteLLMPydanticObjectBase):
    """Body accepted by POST /ptu_reservation/close."""

    id: str = Field(description="Reservation id to close.")
    effective_to: datetime | None = Field(
        default=None,
        description="Timestamp to set as effective_to. Defaults to now (UTC) if omitted.",
    )

    @field_validator("effective_to", mode="after")
    @classmethod
    def _coerce_utc(cls, value: datetime | None) -> datetime | None:
        return _to_utc(value)
