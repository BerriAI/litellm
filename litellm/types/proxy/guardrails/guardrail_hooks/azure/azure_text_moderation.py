from typing import Any, Dict, List, Literal, Optional, Required, TypedDict


class AzureTextModerationRequestBodyOptionalParams(TypedDict, total=False):
    """Optional parameters for the Azure Text Moderation guardrail"""

    categories: Optional[List[str]]
    blocklistNames: Optional[List[str]]
    haltOnBlocklistHit: Optional[bool]
    outputType: Literal["FourSeverityLevels", "EightSeverityLevels"]


class AzureTextModerationGuardrailRequestBody(
    AzureTextModerationRequestBodyOptionalParams
):
    """Configuration parameters for the Azure Text Moderation guardrail"""

    text: Required[str]


class AzureTextModerationGuardrailResponseCategoriesAnalysis(TypedDict):
    """Response from the Azure Text Moderation guardrail"""

    category: str
    severity: int


class AzureTextModerationGuardrailResponse(TypedDict):
    """Response from the Azure Text Moderation guardrail"""

    blocklistsMatch: List[Dict[str, Any]]
    categoriesAnalysis: List[AzureTextModerationGuardrailResponseCategoriesAnalysis]
