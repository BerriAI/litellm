"""Pydantic v2 type definitions for the managed_agents proxy feature."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DockerfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    container_port: int = 4096
    build_platform: str = "linux/amd64"


class AwsOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster: Optional[str] = None
    subnets: Optional[List[str]] = None
    security_group: Optional[str] = None
    task_role_arn: Optional[str] = None
    task_exec_role_arn: Optional[str] = None
    log_group: Optional[str] = None


class ManagedAgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    aws_region: Optional[str] = None
    dockerfiles: Dict[str, DockerfileConfig] = Field(default_factory=dict)
    aws: AwsOverrides = Field(default_factory=AwsOverrides)
    reconcile_interval_seconds: int = 60
    pool_enabled: bool = False
    pool_min_warm: int = 1


class DockerfileOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    container_port: int


class TemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    dockerfile_id: str
    repo_url: str
    default_branch: str
    visibility: str = Field(pattern="^(public|private)$")
    git_token: Optional[str] = None


class TemplateOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: Optional[str] = None
    dockerfile_id: str
    container_port: int
    repo_url: str
    default_branch: str
    visibility: str
    image_uri: Optional[str] = None
    task_def_arn: Optional[str] = None
    build_status: str
    build_error: Optional[str] = None


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    model: str
    prompt: Optional[str] = None
    tools: List[Any] = Field(default_factory=list)
    template_id: str
    branch: Optional[str] = None
    litellm_api_key: Optional[str] = None
    litellm_api_base: Optional[str] = None
    pfp_url: Optional[str] = None
    mcp_servers: List[str] = Field(default_factory=list)


class AgentUpdate(BaseModel):
    """Partial update — every field is optional. Only provided fields are
    written. Setting a string field to "" clears it; null is treated as
    no-op (field absent in the request).

    mcp_servers is replace-style: passing an empty list clears the binding;
    omitting it leaves the existing list untouched.
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    pfp_url: Optional[str] = None
    mcp_servers: Optional[List[str]] = None


class AgentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: Optional[str] = None
    model: str
    prompt: Optional[str] = None
    template_id: str
    branch: str
    pfp_url: Optional[str] = None
    mcp_servers: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class SessionCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_prompt: Optional[str] = None
    title: Optional[str] = None


class SessionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    sandbox_url: Optional[str] = None
    status: str
    task_arn: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = None
