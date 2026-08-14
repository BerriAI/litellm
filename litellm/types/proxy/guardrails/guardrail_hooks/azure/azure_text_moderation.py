from typing import Any, Final, Literal

from pydantic import BaseModel, Field
from typing_extensions import Required, TypedDict

from ..base import GuardrailConfigModel
from .base import AzureContentSafetyConfigModel

# Shared Azure Content Safety categories
AZURE_CONTENT_SAFETY_CATEGORIES: Final = ["Hate", "SelfHarm", "Sexual", "Violence"]


class AzureTextModerationRequestBodyOptionalParams(TypedDict, total=False):
    """Optional parameters for the Azure Text Moderation guardrail"""

    categories: list[str] | None
    blocklistNames: list[str] | None
    haltOnBlocklistHit: bool | None
    outputType: Literal["FourSeverityLevels", "EightSeverityLevels"]


class AzureTextModerationGuardrailRequestBody(AzureTextModerationRequestBodyOptionalParams):
    """Configuration parameters for the Azure Text Moderation guardrail"""

    text: Required[str]


class AzureTextModerationGuardrailResponseCategoriesAnalysis(TypedDict):
    """Response from the Azure Text Moderation guardrail"""

    category: str
    severity: int


class AzureTextModerationGuardrailResponse(TypedDict):
    """Response from the Azure Text Moderation guardrail"""

    blocklistsMatch: list[dict[str, Any]]
    categoriesAnalysis: list[AzureTextModerationGuardrailResponseCategoriesAnalysis]


AzureHarmCategories = Literal["Hate", "SelfHarm", "Sexual", "Violence"]


class AzureTextModerationOptionalParams(BaseModel):
    severity_threshold: int | None = Field(
        default=None,
        description="Severity threshold for the Azure Content Safety Text Moderation guardrail across all categories",
    )
    severity_threshold_by_category: dict[AzureHarmCategories, int] | None = Field(
        default=None,
        description="Severity threshold by category for the Azure Content Safety Text Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )

    categories: list[AzureHarmCategories] | None = Field(
        default=None,
        description="Categories to scan for the Azure Content Safety Text Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )
    blocklistNames: list[str] | None = Field(
        default=None,
        description="Blocklist names to scan for the Azure Content Safety Text Moderation guardrail. Learn more - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text",
    )
    haltOnBlocklistHit: bool | None = Field(
        default=None,
        description="Whether to halt the request if a blocklist hit is detected",
    )
    outputType: Literal["FourSeverityLevels", "EightSeverityLevels"] | None = Field(
        default=None,
        description="Output type for the Azure Content Safety Text Moderation guardrail. Learn more - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text",
    )


class AzureContentSafetyTextModerationConfigModel(
    AzureContentSafetyConfigModel,
    GuardrailConfigModel[AzureTextModerationOptionalParams],
):
    optional_params: AzureTextModerationOptionalParams = Field(
        description="Optional parameters for the Azure Content Safety Text Moderation guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Azure Content Safety Text Moderation"
