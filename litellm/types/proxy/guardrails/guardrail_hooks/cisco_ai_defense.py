"""
Cisco AI Defense Guardrail Config Model
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import GuardrailConfigModel

CISCO_AI_DEFENSE_RULE_NAMES = Literal[
    "Code Detection",
    "Harassment",
    "Hate Speech",
    "PCI",
    "PHI",
    "PII",
    "Prompt Injection",
    "Profanity",
    "Sexual Content & Exploitation",
    "Social Division & Polarization",
    "Violence & Public Safety Threats",
]


# Inspection surfaces supported by Cisco AI Defense. The Cisco Inspection API
# exposes two separate endpoints — one for LLM chat conversations and one for
# MCP tool calls. The user picks exactly one surface to scan per guardrail
# instance; configure two guardrails if you need to scan both.
CISCO_AI_DEFENSE_INSPECTION_TYPE = Literal["chat", "mcp"]


class CiscoAIDefenseRule(BaseModel):
    """A single rule to enable for Cisco AI Defense inspection."""

    rule_name: CISCO_AI_DEFENSE_RULE_NAMES = Field(
        description="The canonical Cisco AI Defense rule name to evaluate.",
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of entity types for the rule (e.g. 'Email Address', "
            "'Phone Number'). Applies to rules such as PII, PCI, and PHI."
        ),
    )


class CiscoAIDefenseGuardrailConfigModelOptionalParams(BaseModel):
    """Optional parameters for the Cisco AI Defense guardrail."""

    model_config = ConfigDict(extra="allow")

    inspection_type: CISCO_AI_DEFENSE_INSPECTION_TYPE = Field(
        default="chat",
        description=(
            "Which Cisco AI Defense inspection surface to use. "
            "'chat' scans LLM model conversations via /api/v1/inspect/chat. "
            "'mcp' scans MCP tool calls via /api/v1/inspect/mcp. "
            "Each guardrail instance targets exactly one surface; configure "
            "two guardrails to scan both chat and MCP traffic."
        ),
    )
    inspect_path: Optional[str] = Field(
        default=None,
        description=(
            "Override for the inspection endpoint path. Defaults to "
            "/api/v1/inspect/chat when inspection_type='chat' and "
            "/api/v1/inspect/mcp when inspection_type='mcp'."
        ),
    )
    enabled_rules: Optional[List[CiscoAIDefenseRule]] = Field(
        default=None,
        description=(
            "Explicit list of Cisco AI Defense rules to evaluate. If omitted, "
            "the policies configured for the API key in the Cisco AI Defense "
            "UI are used."
        ),
    )
    integration_profile_id: Optional[str] = Field(
        default=None,
        description="Integration profile id to apply (advanced).",
    )
    integration_profile_version: Optional[str] = Field(
        default=None,
        description="Integration profile version to apply (advanced).",
    )
    integration_tenant_id: Optional[str] = Field(
        default=None,
        description="Integration tenant id to apply (advanced).",
    )
    integration_type: Optional[str] = Field(
        default=None,
        description="Integration type to apply (advanced).",
    )
    on_flagged_action: Optional[str] = Field(
        default="block",
        description=(
            "Action to take when Cisco AI Defense flags content. 'block' raises "
            "an HTTPException; 'monitor' logs the detection and lets the "
            "request continue."
        ),
    )
    fallback_on_error: Optional[Literal["allow", "block"]] = Field(
        default="block",
        description=(
            "Behaviour when the Cisco AI Defense API is unavailable: 'allow' "
            "proceeds without scanning (high availability), 'block' rejects "
            "the request (maximum security)."
        ),
    )
    timeout: Optional[float] = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout (seconds) for Cisco AI Defense API calls (1-60).",
    )


class CiscoAIDefenseGuardrailConfigModel(GuardrailConfigModel[CiscoAIDefenseGuardrailConfigModelOptionalParams]):
    """Configuration parameters for the Cisco AI Defense guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for the Cisco AI Defense inspection endpoint. If "
            "not provided, the `CISCO_AI_DEFENSE_API_KEY` environment variable "
            "is used. Sent in the `X-Cisco-AI-Defense-API-Key` header. "
            "Both the chat and MCP endpoints use this key."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Regional base URL for the Cisco AI Defense Inspection API. "
            "Defaults to https://us.api.inspect.aidefense.security.cisco.com. "
            "Supported regions: us (us-west-2), ap (ap-ne-1), eu "
            "(eu-central-1). The environment variable "
            "`CISCO_AI_DEFENSE_API_BASE` is consulted as a fallback. The "
            "endpoint path is derived from inspection_type "
            "(/api/v1/inspect/chat for 'chat', /api/v1/inspect/mcp for 'mcp')."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Cisco AI Defense"
