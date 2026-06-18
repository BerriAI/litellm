"""Types for the Sensitive Data Routing guardrail."""

from typing import List, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class SensitiveDataRoutingConfigModel(GuardrailConfigModel):
    """Configuration for the Sensitive Data Routing guardrail."""

    on_premise_model: Optional[str] = Field(
        default=None,
        description="Name of the model group to route to when sensitive data is detected. Must be present in your model_list.",
    )
    prebuilt_patterns: Optional[List[str]] = Field(
        default=None,
        description="Names of built-in detection patterns to match against, e.g. 'us_ssn', 'credit_card', 'email'. See the litellm content filter prebuilt patterns for the full list.",
    )
    regex_patterns: Optional[List[str]] = Field(
        default=None,
        description="Custom regular expressions; a match in any request message reroutes the request on-premise.",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Case-insensitive keywords; presence in any request message reroutes the request on-premise.",
    )
    sticky_session: bool = Field(
        default=True,
        description="When True, once sensitive data is detected in a session every later turn in that session is also routed on-premise, even turns that contain no sensitive data. Requires the client to send a stable session id.",
    )
    session_ttl_seconds: int = Field(
        default=14400,
        description="How long, in seconds, a session stays pinned on-premise after sensitive data is detected.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Sensitive Data Routing"
