from typing import Optional

from pydantic import BaseModel, Field


class AzureContentSafetyConfigModel(BaseModel):
    """Configuration parameters for the Azure Content Safety Prompt Shield guardrail"""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the Azure Content Safety Prompt Shield guardrail",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Azure Content Safety Prompt Shield guardrail",
    )
    api_version: Optional[str] = Field(
        default="2024-09-01",
        description="API version for the Azure Content Safety Prompt Shield guardrail. Default is 2024-09-01",
    )
