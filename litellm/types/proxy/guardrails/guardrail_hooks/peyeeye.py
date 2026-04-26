from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class PEyeEyeGuardrailConfigModelOptionalParams(BaseModel):
    peyeeye_locale: Optional[str] = Field(
        default="auto",
        description="BCP-47 language tag for PII detection. 'auto' lets peyeeye detect.",
    )
    peyeeye_entities: Optional[List[str]] = Field(
        default=None,
        description="Restrict detection to these entity IDs (e.g. ['EMAIL', 'CARD']). Omit to detect all 60+ built-in entities.",
    )
    peyeeye_session_mode: Optional[Literal["stateful", "stateless"]] = Field(
        default="stateful",
        description="'stateful' stores token→value mappings under a peyeeye session id. 'stateless' returns a sealed rehydration key — no PII retained server-side.",
    )


class PEyeEyeGuardrailConfigModel(
    GuardrailConfigModel[PEyeEyeGuardrailConfigModelOptionalParams]
):
    api_key: Optional[str] = Field(
        default=None,
        description="Peyeeye API key. Falls back to the `PEYEEYE_API_KEY` environment variable.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Peyeeye API base URL. Defaults to https://api.peyeeye.ai. Also reads `PEYEEYE_API_BASE`.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Peyeeye PII Redaction & Rehydration"
