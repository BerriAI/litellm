"""VM provider implementations for agent sessions."""

from litellm.proxy.agent_session_endpoints.vm_providers.base import (
    AgentVMProvider,
    ProvisionResult,
)
from litellm.proxy.agent_session_endpoints.vm_providers.noop import NoopVMProvider
from litellm.proxy.agent_session_endpoints.vm_providers.registry import (
    get_vm_provider,
    register_vm_provider,
)

__all__ = [
    "AgentVMProvider",
    "ProvisionResult",
    "NoopVMProvider",
    "get_vm_provider",
    "register_vm_provider",
]
