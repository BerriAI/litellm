from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class SingulrGuardrailRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[list[dict[str, Any]]] = None
    tools: Optional[list[dict[str, Any]]] = None
    model_response: Optional[dict[str, Any]] = None
    litellm_metadata: Optional[dict[str, Any]] = None


class SingulrGuardrailPayload(BaseModel):
    request_data: Optional[SingulrGuardrailRequest] = None
    input_type: str
    is_playground_request: Optional[bool] = None
    playground_text: Optional[str] = None


class SingulrGuardrailResponse(BaseModel):
    """Response returned by the Singulr guardrail API."""

    should_block: bool = False
    blocking_due_to: Optional[str] = None


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    singulr_api_key: Optional[str] = Field(
        default=None,
        description="The Singulr API key. Generate API key from Singulr Platform.",
    )

    singulr_api_base: Optional[str] = Field(
        default=None,
        description="The Singulr API base URL. Get base URL from Singulr Platform.",
    )

    singulr_application_id: Optional[str] = Field(
        default=None,
        description="The Singulr application ID. Get application ID from Singulr Platform.",
    )

    singulr_guardrail_id: Optional[str] = Field(
        default=None,
        description="The Singulr Guardrail ID. Get guardrail ID from Singulr Platform.",
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
