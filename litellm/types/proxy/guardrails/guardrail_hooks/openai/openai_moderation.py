from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..base import GuardrailConfigModel


class BaseOpenAIModerationGuardrailConfigModel(GuardrailConfigModel):
    """Base configuration model for the OpenAI Moderation guardrail"""
    model: Optional[Literal["omni-moderation-latest", "text-moderation-latest"]] = Field(
        default="omni-moderation-latest",
        description="The OpenAI moderation model to use. 'omni-moderation-latest' supports more categorization options and multi-modal inputs. Defaults to 'omni-moderation-latest'.",
    )

class OpenAIModerationGuardrailConfigModel(BaseOpenAIModerationGuardrailConfigModel):
    """Configuration model for the OpenAI Moderation guardrail"""
    
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.",
    )
    
    api_base: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL. Defaults to 'https://api.openai.com/v1'.",
    )

    

    @staticmethod
    def ui_friendly_name() -> str:
        return "OpenAI Moderation" 