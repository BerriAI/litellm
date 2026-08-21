"""Types for the built-in Sensitive Data Routing guardrail."""

from typing import Final

from pydantic import Field

from .base import GuardrailConfigModel

DEFAULT_SESSION_TTL_SECONDS: Final = 14400


class SensitiveDataRoutingGuardrailConfigModel(GuardrailConfigModel):
    """Configuration for the built-in Sensitive Data Routing guardrail."""

    on_premise_model: str | None = Field(
        default=None,
        description="Model group (from model_list) to route the request to when sensitive data is detected.",
    )
    prebuilt_patterns: list[str] | None = Field(
        default=None,
        description="Prebuilt pattern names to match (e.g. us_ssn, credit_card, email).",
    )
    regex_patterns: list[str] | None = Field(
        default=None,
        description="Custom regular expressions; a match in any message reroutes the request.",
    )
    keywords: list[str] | None = Field(
        default=None,
        description="Case-insensitive keywords; a match in any message reroutes the request.",
    )
    sticky_session: bool = Field(
        default=True,
        description="Keep the whole session on the on-premise model after sensitive data is first detected.",
    )
    session_ttl_seconds: int = Field(
        default=DEFAULT_SESSION_TTL_SECONDS,
        ge=1,
        description="How long a session stays pinned to the on-premise model after detection.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Sensitive Data Routing"
