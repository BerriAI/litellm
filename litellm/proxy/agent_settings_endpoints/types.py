"""
Pydantic request/response models for the Cloud Agents settings endpoints.

The split matters:
* `AgentVMConfigUpdateRequest` accepts plaintext AWS keys; the endpoint encrypts
  them before they touch the DB.
* `AgentVMConfigResponse` redacts those keys to a "***" sentinel and returns
  only the metadata. We never decrypt secrets onto a GET response (LIT-2891
  validation #2).
* `AgentSecretCreateRequest` accepts a plaintext value; `AgentSecretResponse`
  has no `value` field at all so the type system itself blocks accidental
  exposure.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

REDACTED_VALUE = "***"

NetworkAccessMode = Literal["allow_all", "allowlist_only"]
AgentSecretType = Literal["env", "file"]
AgentSecretScope = Union[Literal["all"], List[str]]


class NetworkAccessConfig(BaseModel):
    mode: NetworkAccessMode = "allow_all"
    allowlist: List[str] = Field(default_factory=list)


class AgentVMConfigUpdateRequest(BaseModel):
    provider: Optional[Literal["ec2", "self_hosted", "disabled"]] = None
    aws_auth_method: Optional[
        Literal["access_keys", "iam_role", "instance_metadata"]
    ] = None
    aws_access_key_id: Optional[str] = None  # plaintext on the wire
    aws_secret_access_key: Optional[str] = None  # plaintext on the wire
    aws_role_arn: Optional[str] = None  # plaintext on the wire
    aws_region: Optional[str] = None
    ami_id: Optional[str] = None
    instance_type: Optional[str] = None
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    iam_instance_profile: Optional[str] = None
    use_spot: Optional[bool] = None
    max_session_minutes: Optional[int] = Field(default=None, ge=1, le=1440)
    warm_pool_enabled: Optional[bool] = None
    warm_pool_size: Optional[int] = Field(default=None, ge=0, le=100)
    max_idle_minutes: Optional[int] = Field(default=None, ge=0, le=1440)
    hydrate_transport: Optional[Literal["auto", "ssm", "long_poll"]] = None
    network_access: Optional[NetworkAccessConfig] = None
    self_hosted_enabled: Optional[bool] = None


class AgentVMConfigResponse(BaseModel):
    """GET response — AWS creds are redacted to REDACTED_VALUE if set, None if unset."""

    team_id: str
    provider: str
    aws_auth_method: Optional[str]
    aws_access_key_id: Optional[str]  # "***" if set, else None
    aws_secret_access_key: Optional[str]  # "***" if set, else None
    aws_role_arn: Optional[str]  # "***" if set, else None
    aws_region: Optional[str]
    ami_id: Optional[str]
    instance_type: Optional[str]
    subnet_id: Optional[str]
    security_group_id: Optional[str]
    iam_instance_profile: Optional[str]
    use_spot: bool
    max_session_minutes: int
    warm_pool_enabled: bool
    warm_pool_size: int
    max_idle_minutes: int
    hydrate_transport: str
    network_access: NetworkAccessConfig
    self_hosted_enabled: bool


class TestConnectionResponse(BaseModel):
    ok: bool
    account_id: Optional[str] = None
    arn: Optional[str] = None
    region: Optional[str] = None
    error: Optional[str] = None


class AgentSecretCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    value: str = Field(min_length=1)
    scope: AgentSecretScope = "all"
    type: AgentSecretType = "env"
    file_path: Optional[str] = None


class AgentSecretUpdateRequest(BaseModel):
    value: Optional[str] = Field(default=None, min_length=1)
    scope: Optional[AgentSecretScope] = None
    type: Optional[AgentSecretType] = None
    file_path: Optional[str] = None


class AgentSecretResponse(BaseModel):
    """Note: NO `value` field — this type itself enforces validation #2."""

    name: str
    scope: AgentSecretScope
    type: AgentSecretType
    file_path: Optional[str]
    created_at: str
    updated_at: str


class AgentSecretListResponse(BaseModel):
    secrets: List[AgentSecretResponse]


class PairTokenResponse(BaseModel):
    token: str  # raw token returned ONCE — never persisted in DB
    expires_at: str
    install_command: str


class AgentWorkerResponse(BaseModel):
    id: str
    hostname: str
    status: str
    last_seen_at: Optional[str]
    cpu_pct: Optional[float]
    mem_gb: Optional[float]
    active_sessions: int


class AgentWorkerListResponse(BaseModel):
    workers: List[AgentWorkerResponse]


class AgentWorkerRegisterRequest(BaseModel):
    pair_token: str = Field(min_length=1)
    hostname: str = Field(min_length=1, max_length=255)


class AgentWorkerRegisterResponse(BaseModel):
    worker_id: str
    worker_jwt: str  # long-lived JWT, returned ONCE
