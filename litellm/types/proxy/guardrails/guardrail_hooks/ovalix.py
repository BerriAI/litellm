"""Pydantic config model for the Ovalix guardrail (Tracker API, application and checkpoint IDs)."""

from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class OvalixGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Ovalix guardrail (pre/post call checkpoints)."""

    tracker_api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Ovalix Tracker service.",
    )
    tracker_api_key: Optional[str] = Field(
        default=None,
        description="API key for the Ovalix Tracker service.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="Application ID for the Ovalix Tracker service.",
    )
    pre_checkpoint_id: Optional[str] = Field(
        default=None,
        description="Pre-checkpoint ID for the Ovalix Tracker service.",
    )
    post_checkpoint_id: Optional[str] = Field(
        default=None,
        description="Post-checkpoint ID for the Ovalix Tracker service.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        """Display name for this guardrail in the proxy UI."""
        return "Ovalix Guardrail"
