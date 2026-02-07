from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel

class NomaGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security. Also checks if the NOMA_API_KEY env var is set.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security"
