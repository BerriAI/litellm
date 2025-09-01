from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class ModelArmorGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for Google Cloud Model Armor guardrail"""

    template_id: Optional[str] = Field(
        default=None, description="The ID of your Model Armor template"
    )
    project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID"
    )
    location: Optional[str] = Field(
        default=None, description="Google Cloud location/region (e.g., us-central1)"
    )
    credentials: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud credentials JSON file or JSON string",
    )
    api_endpoint: Optional[str] = Field(
        default=None, description="Optional custom API endpoint for Model Armor"
    )
    fail_on_error: Optional[bool] = Field(
        default=True,
        description="Whether to fail the request if Model Armor encounters an error",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        """Return the UI-friendly name for Model Armor guardrail"""
        return "Google Cloud Model Armor"