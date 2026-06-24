"""
Author: Madan Singhal
Date: 23/06/26

"""

from typing import Optional
from pydantic import Field
from .base import GuardrailConfigModel


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate requests to the Singulr Guardrails API.",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Singulr Guardrails API.",
    )

    enforcement_entity_id: Optional[str] = Field(
        default=None,
        description="Identifier of the Singulr enforcement entity used for guardrail evaluation.",
    )

    guardrail_id: Optional[str] = Field(
        default=None,
        description="Identifier of the Singulr guardrail configuration to apply.",
    )

    block_on_error: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to block requests when the Singulr Guardrails API is unavailable "
            "or returns an error. If enabled, requests fail closed. "
            "If disabled, requests continue without guardrail enforcement (fail open)."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Singulr"
