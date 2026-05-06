"""
Pluggable VM providers for `agent_session_endpoints`.

Public re-exports:
- `AgentVMProvider` — the ABC every provider implements
- `ProvisionContext` / `VMHandle` / `VMStatus` / `VMState` — typed I/O
- `AwsCreds` / `Ec2Config` — BYOC inputs
- `get_vm_provider` — registry/factory keyed off provider name
- `register_vm_provider` / `reset_vm_provider_registry` — test helpers
- `ProvisionError` / `InvalidCredentialsError` — error types
"""

from litellm.proxy.agent_session_endpoints.vm_providers.base import (
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
from litellm.proxy.agent_session_endpoints.vm_providers.ec2 import EC2Provider
from litellm.proxy.agent_session_endpoints.vm_providers.factory import (
    SUPPORTED_PROVIDERS,
    build_vm_provider,
)
from litellm.proxy.agent_session_endpoints.vm_providers.noop import NoopProvider
from litellm.proxy.agent_session_endpoints.vm_providers.registry import (
    get_vm_provider,
    register_vm_provider,
    reset_vm_provider_registry,
)

# Backward-compat alias — Epic A's session_endpoints + tests import the noop
# under the older `NoopVMProvider` name.
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
