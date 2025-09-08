from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class PangeaGuardrailConfigModelOptionalParams(BaseModel):
    pangea_input_recipe: Optional[str] = Field(
        default=None,
        description="The Pangea input recipe for the Pangea guardrail. Used for pre-call hook.",
    )
    pangea_output_recipe: Optional[str] = Field(
        default=None,
        description="The Pangea output recipe for the Pangea guardrail. Used for post-call hook.",
    )


class PangeaGuardrailConfigModel(
    GuardrailConfigModel[PangeaGuardrailConfigModelOptionalParams]
):
    api_key: Optional[str] = Field(
        default=None,
        description="The Pangea API key. Reads from PANGEA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Pangea API base URL. Defaults to https://ai-guard.aws.us.pangea.cloud. Also checks if the PANGEA_API_BASE env var is set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Pangea Guardrail"
