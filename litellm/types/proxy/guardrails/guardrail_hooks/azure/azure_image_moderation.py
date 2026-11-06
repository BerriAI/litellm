from typing import Final, Literal

from pydantic import Field
from typing_extensions import Required, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel

from ..base import GuardrailConfigModel
from .azure_text_moderation import AzureHarmCategories
from .base import AzureCategoriesAnalysis, AzureContentSafetyConfigModel

AZURE_CONTENT_SAFETY_IMAGE_CATEGORIES: Final = ["Hate", "SelfHarm", "Sexual", "Violence"]


class AzureImageModerationImage(TypedDict):
    """One base64 image for the Azure Content Safety image analysis endpoint

    Remote URLs are rejected by the guardrail because the bytes Azure scans and
    the bytes the model later fetches can differ.
    """

    content: str


class AzureImageModerationGuardrailRequestBody(TypedDict, total=False):
    """Request body for the Azure Content Safety image analysis endpoint"""

    image: Required[AzureImageModerationImage]
    categories: list[str]
    outputType: Literal["FourSeverityLevels"]


class AzureImageModerationGuardrailResponse(TypedDict):
    """Response from the Azure Content Safety image analysis endpoint"""

    categoriesAnalysis: list[AzureCategoriesAnalysis]


class AzureImageModerationOptionalParams(LiteLLMBaseModel):
    severity_threshold: int | None = Field(
        default=None,
        description="Severity threshold for the Azure Content Safety Image Moderation guardrail across all categories",
    )
    severity_threshold_by_category: dict[AzureHarmCategories, int] | None = Field(
        default=None,
        description="Severity threshold by category for the Azure Content Safety Image Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )
    categories: list[AzureHarmCategories] | None = Field(
        default=None,
        description="Categories to scan for the Azure Content Safety Image Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )


class AzureContentSafetyImageModerationConfigModel(
    AzureContentSafetyConfigModel,
    GuardrailConfigModel[AzureImageModerationOptionalParams],
):
    optional_params: AzureImageModerationOptionalParams | None = Field(
        default=None,
        description="Optional parameters for the Azure Content Safety Image Moderation guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Azure Content Safety Image Moderation"
