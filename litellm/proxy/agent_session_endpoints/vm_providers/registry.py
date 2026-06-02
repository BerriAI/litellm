"""Process-wide registry of VM providers, keyed by provider name."""

from typing import Dict

from litellm.proxy.agent_session_endpoints.vm_providers.base import AgentVMProvider
from litellm.proxy.agent_session_endpoints.vm_providers.noop import NoopVMProvider

_REGISTRY: Dict[str, AgentVMProvider] = {}


def register_vm_provider(provider: AgentVMProvider) -> None:
    """Register a provider by ``provider.name``. Last-write-wins; tests
    use this to swap in a fresh ``NoopVMProvider`` between cases."""
    _REGISTRY[provider.name] = provider


def get_vm_provider(name: str) -> AgentVMProvider:
    """Return the registered provider for ``name``.

    Lazily instantiates a default ``NoopVMProvider`` on first access so
    tests don't need a setup hook just to use the noop.
    """
    if name not in _REGISTRY:
        if name == "noop":
            _REGISTRY[name] = NoopVMProvider()
        else:
            raise KeyError(f"No VM provider registered for '{name}'")
    return _REGISTRY[name]


def reset_vm_provider_registry() -> None:
    """Test helper: drop all registered providers."""
    _REGISTRY.clear()
