from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class GenericGuardrailAPIOptionalParams(BaseModel):
    """Optional parameters for the Generic Guardrail API"""

    additional_provider_specific_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific parameters to send with the guardrail request",
    )


class GenericGuardrailAPIConfigModel(
    GuardrailConfigModel[GenericGuardrailAPIOptionalParams],
):
    """Configuration parameters for the Generic Guardrail API guardrail"""

    optional_params: Optional[GenericGuardrailAPIOptionalParams] = Field(
        default_factory=GenericGuardrailAPIOptionalParams,
        description="Optional parameters for the Generic Guardrail API guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Generic Guardrail API"
