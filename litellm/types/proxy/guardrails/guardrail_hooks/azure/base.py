from pydantic import BaseModel, Field


class AzureContentSafetyConfigModel(BaseModel):
    """Configuration parameters for the Azure Content Safety Prompt Shield guardrail"""

    api_key: str | None = Field(
        default=None,
        description=(
            "API key for the Azure Content Safety resource. Optional: omit it to authenticate with "
            "Microsoft Entra ID, which needs the Cognitive Services User role on the resource and a "
            "custom subdomain api_base"
        ),
    )

    api_base: str | None = Field(
        default=None,
        description="Base URL for the Azure Content Safety Prompt Shield guardrail",
    )
    api_version: str | None = Field(
        default="2024-09-01",
        description="API version for the Azure Content Safety Prompt Shield guardrail. Default is 2024-09-01",
    )
