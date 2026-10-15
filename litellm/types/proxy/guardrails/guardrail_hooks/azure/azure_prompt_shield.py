from typing import Any

from pydantic import Field
from typing_extensions import TypedDict

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

from .base import AzureContentSafetyConfigModel


class AzurePromptShieldGuardrailRequestBody(TypedDict):
    """Configuration parameters for the Azure Prompt Shield guardrail"""

    userPrompt: str
    documents: list[str]


class UserPromptAnalysis(TypedDict, total=False):
    attackDetected: bool


class AzurePromptShieldGuardrailResponse(TypedDict):
    """Configuration parameters for the Azure Prompt Shield guardrail"""

    userPromptAnalysis: UserPromptAnalysis
    documentsAnalysis: list[dict[str, Any]]


class AzurePromptShieldGuardrailConfigModel(
    AzureContentSafetyConfigModel,
    GuardrailConfigModel,
):
    cost_tier: str | None = Field(
        default=None,
        description=(
            "Billing tier of the Azure Content Safety resource: 'free' reports usage with cost 0, "
            "'paid' prices usage with price_per_1000_text_records (required for 'paid'). "
            "Omit to track usage without a cost estimate"
        ),
    )
    price_per_1000_text_records: float | None = Field(
        default=None,
        description=(
            "USD price per 1,000 text records (1 text record = 1,000 characters) used to estimate "
            "Prompt Shield cost. 0 marks the free tier; omit to track usage without a cost estimate"
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Azure Content Safety Prompt Shield"
