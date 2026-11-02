import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import GuardrailConfigModel


def validate_reco_tenant_id(value: str) -> str:
    try:
        uuid.UUID(value)
    except ValueError as e:
        raise ValueError(f"reco_tenant_id must be a valid UUID, got {value!r}") from e
    return value


class RecoOptionalParams(BaseModel):
    """Configuration parameters for the Reco guardrail"""

    model_config = ConfigDict(frozen=True)

    reco_tenant_id: str = Field(
        description="Tenant identifier for the Reco account, as a UUID. Sent as the X-Reco-Tenant-Id header on every guardrail request.",
    )
    api_base: str = Field(
        description="Base URL for the Reco guardrail API. Reco's endpoint is per-region and per-silo, so this must be set explicitly rather than defaulted.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="Behavior when Reco is unreachable. 'fail_closed' (default) blocks the request; 'fail_open' lets it through uninspected. Only applies while fail_on_error is true.",
    )
    fail_on_error: bool = Field(
        default=True,
        description="Behavior on any Reco error, including unreachability. True (default) blocks the request; False lets it through uninspected.",
    )

    @field_validator("reco_tenant_id")
    @classmethod
    def _check_reco_tenant_id(cls, v: str) -> str:
        return validate_reco_tenant_id(v)


class RecoConfigModel(GuardrailConfigModel[RecoOptionalParams]):
    """Configuration parameters for the Reco guardrail"""

    @staticmethod
    def ui_friendly_name() -> str:
        return "Reco"
