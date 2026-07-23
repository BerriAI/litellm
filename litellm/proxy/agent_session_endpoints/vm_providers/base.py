"""
Base abstraction for the agent VM provider.

Each session in `agent_session_endpoints` provisions one VM via a provider that
implements `AgentVMProvider`. The v1 implementation is `EC2Provider` (in
`ec2.py`); `NoopProvider` exists for tests and config-driven swap.

The abstraction takes BYOC AWS creds inside `ProvisionContext` so the proxy
holds no AWS creds itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class Repo:
    """Single git repo to clone into the VM."""

    url: str
    ref: Optional[str] = None  # branch / tag / sha
    path: Optional[str] = None  # mount path inside the VM


@dataclass
class AwsCreds:
    """
    BYOC AWS credentials for a team.

    These are decrypted at use, never logged. The provider passes them straight
    to `boto3.Session(...)` and then drops the reference.
    """

    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region: str = "us-west-2"

    def __repr__(self) -> str:
        # Defensive __repr__ so we cannot accidentally print creds in logs.
        return (
            f"AwsCreds(region={self.region!r}, access_key_id=***REDACTED***, "
            f"secret_access_key=***REDACTED***, "
            f"session_token={'***' if self.session_token else None})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Ec2Config:
    """Per-team EC2 overrides resolved from `LiteLLM_AgentVMConfig`."""

    region: str = "us-west-2"
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    iam_instance_profile: Optional[str] = None
    instance_type: str = "t3.large"
    use_spot: bool = True
    ami_id: Optional[str] = None  # falls back to `default_ami_id` in config.yaml


@dataclass
class ProvisionContext:
    """Inputs to `AgentVMProvider.provision`."""

    session_id: str
    team_id: str
    agent_id: Optional[str] = None
    repos: List[Repo] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)  # from Epic G
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    aws_creds: Optional[AwsCreds] = None  # BYOC; required for the EC2 provider
    ec2_config: Optional[Ec2Config] = None
    # Daemon callback fields. Populated by the session-create endpoint.
    daemon_jwt: Optional[str] = None
    daemon_base_url: Optional[str] = None
    # `session` for cold-boot, `warm` for warm-pool prewarming (B2).
    mode: str = "session"


class VMState(str, Enum):
    """Coarse VM lifecycle states surfaced to the rest of the proxy."""

    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass
class VMHandle:
    """
    Opaque handle returned by `provision`.

    `vm_id` is the provider-native id (EC2 instance id, etc.). `metadata`
    carries the per-provider state we need for `terminate` / `status` (region,
    purchase mode, public IP if any).
    """

    vm_id: str
    provider: str  # `noop`, `ec2`, ...
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VMStatus:
    """Result of `AgentVMProvider.status`."""

    state: VMState
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class AgentVMProvider(ABC):
    """
    Pluggable provider for per-session VMs.

    Implementations must be safe to call concurrently. Implementations MUST NOT
    log AWS credentials; the only place creds enter is via the `AwsCreds` field
    on `ProvisionContext`.
    """

    name: str  # set by subclass: `noop`, `ec2`, ...

    @abstractmethod
    async def provision(self, ctx: ProvisionContext) -> VMHandle:
        """Create a new VM. Must be idempotent on `ctx.session_id`."""

    @abstractmethod
    async def terminate(self, vm: VMHandle) -> None:
        """Terminate the VM. Must be idempotent (no-op if already terminated)."""

    @abstractmethod
    async def status(self, vm: VMHandle) -> VMStatus:
        """Return the current VM status."""


class ProvisionError(Exception):
    """Raised when provisioning fails. Carries an HTTP-mappable status hint."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class InvalidCredentialsError(ProvisionError):
    """BYOC creds are missing, expired, or rejected by the provider."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)
