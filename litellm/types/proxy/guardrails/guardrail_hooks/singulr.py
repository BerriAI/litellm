from typing import Any

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class SingulrGuardrailRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    model_response: dict[str, Any] | None = None
    litellm_metadata: dict[str, Any] | None = None


class SingulrGuardrailPayload(BaseModel):
    litellm_call_id: str | None = None
    request_data: SingulrGuardrailRequest | None = None
    input_type: str
    is_playground_request: bool | None = None
    playground_text: str | None = None


class SingulrGuardrailResponse(BaseModel):
    """Response returned by the Singulr guardrail API."""

    should_block: bool = False
    blocking_due_to: str | None = None


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    singulr_api_key: str | None = Field(
        default=None,
        description="The Singulr API key. Generate API key from Singulr Platform.",
    )

    singulr_api_base: str | None = Field(
        default=None,
        description="The Singulr API base URL. Get base URL from Singulr Platform.",
    )

    singulr_application_id: str | None = Field(
        default=None,
        description="The Singulr application ID. Get application ID from Singulr Platform.",
    )

    singulr_guardrail_id: str | None = Field(
        default=None,
        description="The Singulr Guardrail ID. Get guardrail ID from Singulr Platform.",
    )

    block_on_error: bool | None = Field(
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
