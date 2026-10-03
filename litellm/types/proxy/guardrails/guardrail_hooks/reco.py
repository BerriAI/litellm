import uuid

from pydantic import BaseModel, Field, field_validator

from .base import GuardrailConfigModel


def validate_reco_tenant_id(value: str) -> str:
    try:
        uuid.UUID(value)
    except ValueError as e:
        raise ValueError(f"reco_tenant_id must be a valid UUID, got {value!r}") from e
    return value


class RecoOptionalParams(BaseModel):
    """Configuration parameters for the Reco guardrail"""

    reco_tenant_id: str = Field(
        description="Tenant identifier for the Reco account, as a UUID. Sent as the X-Reco-Tenant-Id header on every guardrail request.",
    )
    api_base: str = Field(
        description="Base URL for the Reco guardrail API. Reco's endpoint is per-region and per-silo, so this must be set explicitly rather than defaulted.",
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
