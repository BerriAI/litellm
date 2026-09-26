from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

from .base import AzureContentSafetyConfigModel


class AzurePromptShieldGuardrailRequestBody(TypedDict):
    """Body of one ``text:shieldPrompt`` request: the user's own text plus the
    documents (attachments and tool outputs) Azure analyzes for injected instructions"""

    userPrompt: NotRequired[ReadOnly[str]]
    documents: NotRequired[ReadOnly[Sequence[str]]]


class AzurePromptShieldAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    attackDetected: bool = False


class AzurePromptShieldGuardrailResponse(BaseModel):
    """Parsed ``text:shieldPrompt`` response; ``documentsAnalysis`` follows the order
    of the submitted documents"""

    model_config = ConfigDict(frozen=True)

    userPromptAnalysis: AzurePromptShieldAnalysis | None = None
    documentsAnalysis: tuple[AzurePromptShieldAnalysis, ...] = ()


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
