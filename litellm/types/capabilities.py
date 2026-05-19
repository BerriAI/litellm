"""
Capability Provider response types.

Used by ``GET /v1/capabilities`` and ``GET /.well-known/xct-capabilities``
to expose what an authenticated caller (or anonymous client, for the
public variant) is allowed to use across four entity classes:
model, agent, mcp, skill — plus the access-group bundles that group them.

Design notes (decided in S1-01 ADR):
- snake_case throughout, matching the rest of the LiteLLM proxy.
- Each entity is a *summary*: id + a small set of fields needed to render
  a discovery list. Full detail is fetched per-entity via the existing
  endpoints (`/v1/agents/{id}`, `/v1/mcp/server/{id}`, etc).
- ``caller`` describes who the response is scoped to so the SDK can sanity
  check (and so the same JSON is debuggable in logs).
- The shape is additive-only: future fields are appended; downstream code
  must tolerate unknown fields. Do not rename or remove fields without
  bumping the response ``schema_version``.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from litellm.types.llms.base import LiteLLMPydanticObjectBase


CAPABILITIES_SCHEMA_VERSION = "2026-05-19"


class CapabilityCaller(LiteLLMPydanticObjectBase):
    """Who the capabilities are scoped to."""

    key_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    app_id: Optional[str] = None
    is_admin: bool = False


class ModelCapabilityFlags(BaseModel):
    """Boolean feature flags for a model.

    Derived from ``litellm.utils.get_supported_openai_params`` and
    ``get_model_info``; populated server-side so downstream apps don't
    have to maintain their own provider feature matrix.
    """

    vision: bool = False
    function_calling: bool = False
    structured_output: bool = False
    prompt_caching: bool = False
    pdf_input: bool = False
    web_search: bool = False
    audio_input: bool = False
    audio_output: bool = False


class ModelSummary(LiteLLMPydanticObjectBase):
    """One model entry in the capabilities response."""

    id: str = Field(..., description="Model name (e.g. 'gpt-4o', 'deepseek-v3.2').")
    provider: Optional[str] = None
    mode: Optional[str] = Field(
        None, description="'chat' | 'embedding' | 'image_generation' | …"
    )
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: ModelCapabilityFlags = Field(default_factory=ModelCapabilityFlags)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None


class AgentSummary(LiteLLMPydanticObjectBase):
    """One agent entry in the capabilities response."""

    agent_id: str
    agent_name: str
    description: Optional[str] = None
    version: Optional[str] = None
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    supports_streaming: bool = False
    agent_card_url: Optional[str] = Field(
        None,
        description="Path to .well-known/agent-card.json for full A2A discovery.",
    )


class McpSummary(LiteLLMPydanticObjectBase):
    """One MCP server entry in the capabilities response."""

    server_id: str
    server_name: Optional[str] = None
    alias: Optional[str] = None
    transport: Optional[str] = Field(None, description="'http' | 'sse' | 'stdio'")
    tools_count: Optional[int] = None
    access_groups: List[str] = Field(default_factory=list)
    auth_type: Optional[str] = None
    needs_oauth: bool = False


class SkillSummary(LiteLLMPydanticObjectBase):
    """One skill entry in the capabilities response."""

    skill_id: str
    display_title: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    source: Optional[str] = Field(
        None, description="'xct' | 'anthropic' | 'custom' | …"
    )
    category: Optional[str] = None
    is_public: bool = False


class AccessGroupSummary(LiteLLMPydanticObjectBase):
    """A bundle of models + agents + mcps grantable as one unit."""

    id: str
    name: str
    description: Optional[str] = None
    model_count: int = 0
    agent_count: int = 0
    mcp_count: int = 0


class CapabilitiesResponse(LiteLLMPydanticObjectBase):
    """Top-level response for ``GET /v1/capabilities``."""

    schema_version: str = CAPABILITIES_SCHEMA_VERSION
    caller: CapabilityCaller
    models: List[ModelSummary] = Field(default_factory=list)
    agents: List[AgentSummary] = Field(default_factory=list)
    mcps: List[McpSummary] = Field(default_factory=list)
    skills: List[SkillSummary] = Field(default_factory=list)
    access_groups: List[AccessGroupSummary] = Field(default_factory=list)
    # extension point — downstream tools may stash app-specific hints here
    extensions: Dict[str, Any] = Field(default_factory=dict)


class PublicCapabilitiesResponse(LiteLLMPydanticObjectBase):
    """Public (unauthenticated) variant for ``/.well-known/xct-capabilities``.

    Strips all caller-scoped fields and all credential metadata. Only includes
    entities flagged ``is_public=true`` (or, for models, those listed in the
    public-model allowlist).
    """

    schema_version: str = CAPABILITIES_SCHEMA_VERSION
    models: List[ModelSummary] = Field(default_factory=list)
    agents: List[AgentSummary] = Field(default_factory=list)
    mcps: List[McpSummary] = Field(default_factory=list)
    skills: List[SkillSummary] = Field(default_factory=list)
