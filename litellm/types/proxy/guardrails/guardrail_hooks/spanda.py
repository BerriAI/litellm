from pydantic import Field

from .base import GuardrailConfigModel


class SpandaGuardrailConfigModel(GuardrailConfigModel):
    uncertainty_threshold: float | None = Field(
        default=0.35,
        description="Epistemic uncertainty threshold (R_sc). Responses with R_sc exceeding this threshold are flagged.",
    )
    grounding_threshold: float | None = Field(
        default=0.15,
        description="Grounding residual threshold for Tier-2 hallucination and mode collapse detection.",
    )
    block_mode: bool | None = Field(
        default=False,
        description="If True, blocks responses that exceed the uncertainty threshold or trigger mode collapse.",
    )
    api_base: str | None = Field(
        default=None,
        description="Optional Spanda enterprise gateway proxy URL (e.g. http://localhost:8000). If omitted, runs in-process CPU detection.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Spanda (BRHMN Labs)"
