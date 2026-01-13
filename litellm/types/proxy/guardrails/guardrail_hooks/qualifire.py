from typing import List, Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class QualifireGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Qualifire guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description="The API key for Qualifire. If not provided, the `QUALIFIRE_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base URL for Qualifire. If not provided, the `QUALIFIRE_BASE_URL` environment variable is checked.",
    )
    evaluation_id: Optional[str] = Field(
        default=None,
        description="Pre-configured evaluation ID from Qualifire dashboard. When provided, uses invoke_evaluation() instead of evaluate().",
    )
    prompt_injections: Optional[bool] = Field(
        default=None,
        description="Enable prompt injection detection. Default check if no evaluation_id and no other checks are specified.",
    )
    hallucinations_check: Optional[bool] = Field(
        default=None,
        description="Enable hallucination detection to detect factual inaccuracies.",
    )
    grounding_check: Optional[bool] = Field(
        default=None,
        description="Enable grounding verification to ensure output is grounded in provided context.",
    )
    pii_check: Optional[bool] = Field(
        default=None,
        description="Enable PII (Personally Identifiable Information) detection.",
    )
    content_moderation_check: Optional[bool] = Field(
        default=None,
        description="Enable content moderation to check for harmful content (harassment, hate speech, etc.).",
    )
    tool_selection_quality_check: Optional[bool] = Field(
        default=None,
        description="Enable tool selection quality check to evaluate quality of tool/function calls.",
    )
    assertions: Optional[List[str]] = Field(
        default=None,
        description="Custom assertions to validate against the output. Each assertion is a string describing a condition.",
    )
    on_flagged: Optional[Literal["block", "monitor"]] = Field(
        default="block",
        description="Action to take when content is flagged. 'block' raises an exception, 'monitor' logs but allows the request.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Qualifire"
