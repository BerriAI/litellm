from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class NeedlepathGuardrailOptionalParams(BaseModel):
    """Optional tuning knobs for the Needlepath guardrail."""

    select_tool_outputs: bool | None = Field(
        default=None,
        description="Select over tool/function result messages (search hits, RAG chunks, API dumps). Defaults to True.",
    )
    select_history: bool | None = Field(
        default=None,
        description="Also select over prior (non-last) user messages. Defaults to False.",
    )
    select_system: bool | None = Field(
        default=None,
        description="Also select over system messages. Defaults to False.",
    )
    min_chars_to_select: int | None = Field(
        default=None,
        description="Skip messages whose text is shorter than this many characters. Defaults to 500.",
    )
    max_context_tokens: int | None = Field(
        default=None,
        description="Token budget requested for the selected block of a single message. Defaults to 4000.",
    )
    operating_point: str | None = Field(
        default=None,
        description=(
            "Immutable engine label sent with every request. Defaults to 'np-2026-07-r2'. "
            "Pinned rather than inherited from the service default, so what this guardrail "
            "sends does not change underneath a deployment."
        ),
    )


class NeedlepathGuardrailConfigModel(GuardrailConfigModel[NeedlepathGuardrailOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="Needlepath API key. Falls back to the NEEDLEPATH_API_KEY env var.",
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the Needlepath API. Falls back to the NEEDLEPATH_API_BASE env var, "
            "then https://api.nextmoca.com."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Needlepath (context selection)"
