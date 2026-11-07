from pydantic import Field

from .base import GuardrailConfigModel


class PrivaiteGuardrailConfigModel(GuardrailConfigModel):  # pyright: ignore[reportMissingTypeArgument]  # parameterizing adds an optional_params field to the UI form
    preset: str | None = Field(
        default="onnx",
        description=("Detection preset: 'onnx' (full suite, also detects secrets) or 'light' (faster, Presidio only)."),
    )
    languages: str | None = Field(
        default="en,fr",
        description="Comma-separated spaCy languages used for detection.",
    )
    deanonymize: bool | None = Field(
        default=True,
        description="Restore the original PII values in the response.",
    )
    block_entities: str | tuple[str, ...] | None = Field(
        default=None,
        description=(
            "PII types to reject outright instead of masking: a list, or a "
            "comma-separated string, e.g. ['US_SSN', 'CREDIT_CARD']. A request "
            "containing any listed type is refused with a 400 and nothing is "
            "forwarded. Omitted (the default) masks everything and forwards it."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PrivAiTe"
