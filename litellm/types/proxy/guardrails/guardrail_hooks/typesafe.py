from typing import Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class TypeSafeGuardrailOptionalParams(BaseModel):
    """Optional tuning knobs for the TypeSafe (Jev) compaction guardrail."""

    relevance_threshold: float | None = Field(
        default=None,
        description=(
            "Relevance cutoff in [0, 1]. A completed tool exchange is dropped when Jev "
            "scores the probability that it is still needed below this value. Defaults to 0.2."
        ),
    )
    min_chars_to_evaluate: int | None = Field(
        default=None,
        description=(
            "Skip tool exchanges whose combined tool-result text is shorter than this many characters. Defaults to 200."
        ),
    )
    max_result_chars_in_state: int | None = Field(
        default=None,
        description=(
            "Tool result text is truncated to this many characters when sent to the Jev evaluator. Defaults to 4000."
        ),
    )


class TypeSafeGuardrailConfigModel(GuardrailConfigModel[TypeSafeGuardrailOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="TypeSafe API key, sent as a Bearer token. Falls back to the TYPESAFE_API_KEY env var.",
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the TypeSafe API. Falls back to the TYPESAFE_API_BASE env var, then https://api.typesafe.ai."
        ),
    )
    model: str | None = Field(
        default=None,
        description="TypeSafe evaluation model (not the LLM). Defaults to 'jev-latest'.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_open",
        description=(
            "Behavior when the TypeSafe evaluation service is unreachable or errors. "
            "'fail_open' (default) forwards the request uncompacted. 'fail_closed' "
            "raises an error instead."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "TypeSafe (Jev) Compaction"
