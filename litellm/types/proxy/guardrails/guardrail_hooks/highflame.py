"""Type definitions for the Highflame (Shield) guardrail integration.

Highflame's Shield service exposes a single guard endpoint,
``POST /v1/shield/guard``, that runs the requested detectors and a Cedar
policy evaluation and returns a ``decision``. See
https://docs.highflame.ai for the full contract.
"""

from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel

# ---------------------------------------------------------------------------
# Wire types for POST /v1/shield/guard
# ---------------------------------------------------------------------------


class HighflameSignal(TypedDict, total=False):
    """A single taxonomy-aligned detection signal from Shield."""

    vulnerability_id: str  # taxonomy ID, e.g. "prompt_injection"
    name: str
    severity: str  # low | medium | high | critical
    score: int  # normalized 0-100
    category: str  # taxonomy domain, e.g. "semantic"
    context_key: str


class HighflameGuardRequest(TypedDict, total=False):
    content: str
    content_type: str  # prompt | response | tool_call | file | clipboard
    action: str  # Cedar action, e.g. "process_prompt"
    detectors: List[str]  # Shield detector IDs; empty/omitted = all enabled
    mode: str  # enforce | monitor | alert | modify
    application: str
    metadata: Dict
    session_id: str


class HighflameGuardResponse(TypedDict, total=False):
    decision: str  # allow | deny | alert | modify | monitor | step_up | defer
    policy_reason: str
    signals: List[HighflameSignal]
    redacted_content: Optional[str]
    request_id: str
    latency_ms: int


class HighflameTokenResponse(TypedDict, total=False):
    """Response from the AuthN token-exchange endpoint."""

    access_token: str
    expires_in: int
    account_id: str
    project_id: str
    gateway_id: str


# ---------------------------------------------------------------------------
# Capability surface
# ---------------------------------------------------------------------------
#
# Highflame presents guardrail capabilities in OWASP LLM Top 10 (2025)
# terminology, mapped to the underlying Shield detector IDs. This mirrors
# Highflame's published taxonomy (https://docs.highflame.ai). Users set
# ``capabilities: [...]`` in their guardrail config; an empty/omitted list
# means "apply every guardrail enabled in the Highflame application policy".
HIGHFLAME_CAPABILITY_MAP: Dict[str, List[str]] = {
    # OWASP LLM01 — Prompt Injection
    "prompt_injection": ["injection"],
    # OWASP LLM02 — Sensitive Information Disclosure
    "sensitive_information_disclosure": ["pii", "pii_model", "dlp", "secrets"],
    # OWASP LLM06 — Excessive Agency (agentic / tool safety)
    "excessive_agency": [
        "tool_risk",
        "mcp_risk",
        "tool_poisoning",
        "command_injection",
        "sql_injection",
        "path_traversal",
    ],
    # OWASP LLM09 — Misinformation
    "misinformation": ["hallucination"],
    # OWASP LLM10 — Unbounded Consumption
    "unbounded_consumption": ["budget_checker", "loop_detector"],
    # Trust & safety / responsible-AI content controls (beyond the LLM Top 10)
    "content_safety": ["content_safety", "toxicity"],
    # Utility
    "language_detection": ["language"],
}


class HighflameGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Highflame (Shield) guardrail."""

    capabilities: Optional[List[str]] = Field(
        default=None,
        description=(
            "OWASP-aligned guardrail capabilities to run, e.g. "
            "['prompt_injection', 'sensitive_information_disclosure']. "
            "Empty/omitted runs every guardrail enabled in the Highflame "
            "application policy."
        ),
    )
    application: Optional[str] = Field(
        default=None,
        description="Highflame application name for policy-scoped guardrails.",
    )
    shield_mode: Optional[str] = Field(
        default="enforce",
        description="Shield evaluation mode: enforce | monitor | alert | modify.",
    )
    token_url: Optional[str] = Field(
        default=None,
        description=(
            "OAuth token-exchange URL. Defaults to "
            "https://auth.highflame.ai/oauth2/token."
        ),
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata passed through to Shield detectors.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Highflame Guardrails"
