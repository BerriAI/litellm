"""Pydantic v2 types for managed agents v2.

Wire format: snake_case JSON. Timestamps ISO 8601 UTC.
IDs: prefixed UUIDs — `agt_<uuid4>`, `ses_<uuid4>`, `msg_<uuid4>`.

Request models forbid extra fields; response models allow extras for
forward-compatibility with proxy-side enrichment.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Status / sandbox enums
# ---------------------------------------------------------------------------

SessionStatus = Literal["provisioning", "ready", "terminated", "error"]
MessageStatus = Literal["in_progress", "completed", "failed"]
MessageRole = Literal["user", "assistant"]
SandboxType = Literal["opencode"]
SandboxSize = Literal["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Inner config object on an agent.

    The `litellm_api_key` is masked on read by `masking.mask_litellm_api_key`.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    system_prompt: str
    tools: List[str]
    litellm_api_key: str
    litellm_base_url: str


class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    config: AgentConfig


class AgentRow(BaseModel):
    """Response shape for `POST /v2/agents`. `config.litellm_api_key` is masked."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    config: AgentConfig
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AgentList(BaseModel):
    """Paginated list response for `GET /v2/agents`.

    `config.litellm_api_key` is masked on each row in `data`.
    """

    model_config = ConfigDict(extra="allow")

    data: List[AgentRow]
    next_cursor: Optional[str] = None
    has_more: bool = False


# ---------------------------------------------------------------------------
# Sandbox / session
# ---------------------------------------------------------------------------


class SandboxSpec(BaseModel):
    """Per-session sandbox configuration.

    `image` is optional and falls back to a pinned built-in.
    `timeout_minutes` is bounded to [1, 1440].
    `idle_timeout_minutes` must be in [1, timeout_minutes].
    """

    model_config = ConfigDict(extra="forbid")

    type: SandboxType
    size: SandboxSize = "small"
    timeout_minutes: int = Field(default=60, ge=1, le=1440)
    idle_timeout_minutes: int = Field(default=10, ge=1)
    image: Optional[str] = None

    @model_validator(mode="after")
    def _validate_idle_timeout_within_timeout(self) -> "SandboxSpec":
        """Enforce contract §6.2: `idle_timeout_minutes` ∈ [1, `timeout_minutes`]."""
        if self.idle_timeout_minutes > self.timeout_minutes:
            raise ValueError(
                f"idle_timeout_minutes ({self.idle_timeout_minutes}) must be "
                f"<= timeout_minutes ({self.timeout_minutes})"
            )
        return self


class Repo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    starting_ref: str
    checked_out_sha: Optional[str] = None


class CreateSessionRequest(BaseModel):
    """For `POST /v2/sessions` (Krrish, LIT-2919).

    Wave 1 doesn't implement the endpoint — the type is here for Wave 2/3
    handlers and adapter integration tests.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    sandbox: SandboxSpec
    repos: List[Repo] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)


class SessionRow(BaseModel):
    """Response shape for session reads.

    `sandbox_url` and `sandbox_metadata` are stored on the row internally but
    NEVER returned in the API response — see contract §6.2.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    agent_id: str
    sandbox: SandboxSpec
    status: SessionStatus
    repos: List[Repo] = Field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime
    terminated_at: Optional[datetime] = None


class SessionList(BaseModel):
    """Paginated list response for `GET /v2/agents/:id/sessions`."""

    model_config = ConfigDict(extra="allow")

    data: List[SessionRow]
    next_cursor: Optional[str] = None
    has_more: bool = False


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class CreateMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    model: Optional[str] = None  # falls back to agent.config.model


class MessageRow(BaseModel):
    """Normalized message shape, mapped from the underlying sandbox provider.

    `tools` is omitted when there are no tool calls (assistant-only field).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    session_id: str
    role: MessageRole
    content: str
    model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    status: MessageStatus
    created_at: datetime
    completed_at: Optional[datetime] = None


class MessageList(BaseModel):
    """Paginated list response for `GET /v2/sessions/:id/messages`."""

    model_config = ConfigDict(extra="allow")

    data: List[MessageRow]
    next_cursor: Optional[str] = None
    has_more: bool = False
