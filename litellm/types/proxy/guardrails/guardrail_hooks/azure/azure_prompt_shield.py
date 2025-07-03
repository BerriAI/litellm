from typing import Any, Dict, List, TypedDict


class AzurePromptShieldGuardrailRequestBody(TypedDict):
    """Configuration parameters for the Azure Prompt Shield guardrail"""

    userPrompt: str
    documents: List[str]


class UserPromptAnalysis(TypedDict, total=False):
    attackDetected: bool


class AzurePromptShieldGuardrailResponse(TypedDict):
    """Configuration parameters for the Azure Prompt Shield guardrail"""

    userPromptAnalysis: UserPromptAnalysis
    documentsAnalysis: List[Dict[str, Any]]
