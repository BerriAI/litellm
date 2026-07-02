from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class DataFogGuardrailOptionalParams(BaseModel):
    datafog_entity_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "Entity types to detect. Defaults to the high-precision set: "
            "EMAIL, PHONE, CREDIT_CARD, SSN. Additional types (IP_ADDRESS, "
            "DOB, ZIP, and DE_* locale entities) are opt-in because they "
            "are noisy in technical text."
        ),
    )
    datafog_locales: Optional[list[str]] = Field(
        default=None,
        description=(
            'Locale packs to enable for country-specific entities, e.g. ["de"] for German VAT IDs, IBANs, and tax IDs.'
        ),
    )
    datafog_fail_policy: Optional[Literal["open", "closed"]] = Field(
        default="open",
        description=(
            "Behavior when the detection engine errors. 'open' (default) "
            "lets traffic through unscanned so a guardrail bug never takes "
            "down the gateway; 'closed' rejects traffic instead, for "
            "compliance deployments where unscanned egress is worse than "
            "downtime."
        ),
    )


class DataFogGuardrailConfigModel(GuardrailConfigModel[DataFogGuardrailOptionalParams]):
    datafog_action: Optional[Literal["redact", "block"]] = Field(
        default="redact",
        description=(
            "'redact' (default) replaces detected PII with [TYPE_N] tokens "
            "before the request leaves the gateway; 'block' rejects the "
            "request with an HTTP 400 listing entity-type counts only."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "DataFog PII Guardrail"
