"""
Pluggable VM providers for `agent_session_endpoints`.

Public re-exports:
- `AgentVMProvider` — the ABC every provider implements
- `ProvisionContext` / `VMHandle` / `VMStatus` / `VMState` — typed I/O
- `AwsCreds` / `Ec2Config` — BYOC inputs
- `build_vm_provider` — config-driven factory (called once at startup)
- `get_vm_provider` — runtime registry lookup keyed by provider name
- `register_vm_provider` / `reset_vm_provider_registry` — registry helpers
- `ProvisionError` / `InvalidCredentialsError` — error types
"""

from litellm.managed_agents.vms.base import (
    AgentVMProvider,
    AwsCreds,
    Ec2Config,
    InvalidCredentialsError,
    ProvisionContext,
    ProvisionError,
    Repo,
    VMHandle,
    VMState,
    VMStatus,
)
from litellm.managed_agents.vms.ec2 import EC2Provider
from litellm.managed_agents.vms.factory import (
    SUPPORTED_PROVIDERS,
    build_vm_provider,
)
from litellm.managed_agents.vms.noop import NoopProvider
from litellm.managed_agents.vms.registry import (
    get_vm_provider,
    register_vm_provider,
    reset_vm_provider_registry,
)

# Compatibility alias: A1's session_endpoints + tests imported ``NoopVMProvider``
# while B1 named the class ``NoopProvider``. Keep both names exported so neither
# side needs to chase the rename.
NoopVMProvider = NoopProvider

__all__ = [
    "AgentVMProvider",
    "AwsCreds",
    "Ec2Config",
    "EC2Provider",
    "InvalidCredentialsError",
    "NoopProvider",
    "NoopVMProvider",
    "ProvisionContext",
    "ProvisionError",
    "Repo",
    "SUPPORTED_PROVIDERS",
    "VMHandle",
    "VMState",
    "VMStatus",
    "build_vm_provider",
    "get_vm_provider",
    "register_vm_provider",
    "reset_vm_provider_registry",
]
