"""
Pydantic request/response models for the public agent_session_endpoints API.

These models are the canonical wire shape — Epic D (TS SDK) generates types
from this file via ``model_json_schema``. Keep field names and types stable
once shipped.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Repo / env_vars (shared building blocks)
# ---------------------------------------------------------------------------


class RepoSpec(BaseModel):
    """A git repository to clone into the session VM."""

    url: str
    startingRef: Optional[str] = Field(
        default=None,
        description="Branch, tag, or SHA to check out. Defaults to default branch.",
    )

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentCreate(BaseModel):
    name: str
    model: str
    system_prompt: Optional[str] = None
    default_repos: Optional[List[RepoSpec]] = None
    default_env_vars: Optional[Dict[str, str]] = None
    tools_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    """All fields optional — PATCH semantics. ``None`` means "leave alone"."""

    name: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    default_repos: Optional[List[RepoSpec]] = None
    default_env_vars: Optional[Dict[str, str]] = None
    tools_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    id: str
    name: str
    model: str
    system_prompt: Optional[str] = None
    default_repos: Optional[List[Dict[str, Any]]] = None
    default_env_vars: Optional[Dict[str, str]] = None
    tools_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    team_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    agent_id: str
    repos: Optional[List[RepoSpec]] = None
    env_vars: Optional[Dict[str, str]] = None
    max_session_minutes: Optional[int] = Field(
        default=None,
        description="Override default 4h max. Capped at 24h.",
        ge=1,
        le=24 * 60,
    )


class SessionResponse(BaseModel):
    id: str
    agent_id: str
    status: str
    vm_id: Optional[str] = None
    vm_provider: Optional[str] = None
    repos: List[Dict[str, Any]] = Field(default_factory=list)
    expires_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    terminated_at: Optional[str] = None
    daemon_token: Optional[str] = Field(
        default=None,
        description="Returned ONLY on initial create. Subsequent reads return None.",
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class RunCreate(BaseModel):
    prompt: Dict[str, Any] = Field(
        ...,
        description='Free-form prompt object. ``{"text": "hi"}`` is the minimum.',
    )


class RunResponse(BaseModel):
    id: str
    session_id: str
    status: str
    prompt: Dict[str, Any]
    parent_run_id: Optional[str] = None
    result: Optional[str] = None
    git_branches: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    terminated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Followup
# ---------------------------------------------------------------------------


class FollowupCreate(BaseModel):
    prompt: Dict[str, Any]


class FollowupResponse(BaseModel):
    run_id: str
    action: Literal["queued", "new_run"]


# ---------------------------------------------------------------------------
# Internal endpoints (daemon callbacks)
# ---------------------------------------------------------------------------


class DaemonRegisterRequest(BaseModel):
    vm_id: Optional[str] = None
    daemon_version: Optional[str] = None


class DaemonHeartbeatRequest(BaseModel):
    vm_id: Optional[str] = None


class EventAppend(BaseModel):
    event_type: str
    payload: Dict[str, Any]


class NextRunResponse(BaseModel):
    run_id: str
    prompt: Dict[str, Any]


class ConversationMessage(BaseModel):
    run_id: str
    seq: int
    event_type: str
    payload: Dict[str, Any]
    created_at: Optional[str] = None


class ConversationResponse(BaseModel):
    session_id: str
    messages: List[ConversationMessage]
