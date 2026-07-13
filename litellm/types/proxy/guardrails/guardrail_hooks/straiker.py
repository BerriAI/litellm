from typing import Dict, Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class StraikerGuardrailConfigModel(GuardrailConfigModel):
    api_key: str = Field(
        min_length=1,
        description="Straiker DefendAI API key (Bearer token). Env: STRAIKER_API_KEY.",
        json_schema_extra={"secret": True},
    )

    api_base: str = Field(
        default="https://api.prod.straiker.ai",
        description="Straiker API base URL. Use the regional variant for non-US tenants.",
    )

    agentic: bool = Field(
        default=False,
        description=(
            "When true, posts the full messages[] (with tool_calls + tool results) to "
            "/api/v1/detect for multi-turn / tool-using agents."
        ),
    )

    source: str = Field(
        default="litellm",
        description="Application name registered in the Straiker Defend Console.",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Block when score > threshold. post_call is observability-only and never blocks.",
    )

    timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt HTTP timeout in seconds.",
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries on transient HTTP (408/429/5xx) and network errors.",
    )

    initial_backoff: float = Field(
        default=0.1,
        ge=0.0,
        description="Initial retry backoff in seconds.",
    )

    max_backoff: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum retry backoff in seconds.",
    )

    unreachable_fallback: Literal["fail_open", "fail_closed"] = Field(
        default="fail_closed",
        description="Behavior when Straiker is unreachable after retries.",
    )

    max_payload_bytes: int = Field(
        default=524288,
        gt=0,
        description="Maximum serialized request payload size sent to Straiker.",
    )

    custom_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers sent to Straiker, excluding Authorization.",
    )

    verbose: bool = Field(
        default=False,
        description=(
            "Sends Straiker-Debug header; block responses include the full per-category "
            "detection envelope and triggered_categories."
        ),
    )

    dedup_iterations: bool = Field(
        default=True,
        description=(
            "Suppress assistant-only pre-call and during-call continuations in agentic mode. "
            "User turns, tool results, and completed outputs are still scored."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Straiker"
