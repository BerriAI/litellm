from pydantic import Field

from .base import GuardrailConfigModel


class ModelArmorGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for Google Cloud Model Armor guardrail"""

    template_id: str | None = Field(default=None, description="The ID of your Model Armor template")
    project_id: str | None = Field(default=None, description="Google Cloud project ID")
    location: str | None = Field(default=None, description="Google Cloud location/region (e.g., us-central1)")
    credentials: str | None = Field(
        default=None,
        description="Path to Google Cloud credentials JSON file or JSON string",
    )
    api_endpoint: str | None = Field(default=None, description="Optional custom API endpoint for Model Armor")
    fail_on_error: bool | None = Field(
        default=True,
        description="Whether to fail the request if Model Armor encounters an error",
    )
    sanitize_error_detail: bool | None = Field(
        default=True,
        description=(
            "Omit the raw Model Armor response from caller-facing errors and logs "
            "by default. Set False to restore verbose output."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        """Return the UI-friendly name for Model Armor guardrail"""
        return "Google Cloud Model Armor"
