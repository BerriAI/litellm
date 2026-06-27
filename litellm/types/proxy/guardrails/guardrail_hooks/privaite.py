from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class PrivaiteGuardrailConfigModel(GuardrailConfigModel):
    preset: Optional[str] = Field(
        default="onnx",
        description=("Detection preset: 'onnx' (full suite, also detects secrets) or 'light' (faster, Presidio only)."),
    )
    languages: Optional[str] = Field(
        default="en,fr",
        description="Comma-separated spaCy languages used for detection.",
    )
    deanonymize: Optional[bool] = Field(
        default=True,
        description="Restore the original PII values in the response.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PrivAiTe"
