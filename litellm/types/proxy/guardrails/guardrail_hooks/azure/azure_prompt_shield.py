from typing import Any

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
    @staticmethod
    def ui_friendly_name() -> str:
        return "Azure Content Safety Prompt Shield"
