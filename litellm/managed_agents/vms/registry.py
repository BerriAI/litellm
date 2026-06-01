"""Process-wide registry of VM providers, keyed by provider name.

The registry is the runtime lookup path used by ``session_endpoints.py``,
``sweepers.py``, and tests. At proxy startup we typically build the configured
provider via ``factory.build_vm_provider(agent_settings)`` and then call
``register_vm_provider(provider)`` to install it under its ``provider.name``.

For tests / local-dev convenience, ``get_vm_provider("noop")`` lazily
instantiates a default ``NoopProvider`` on first access so callers don't need
a setup hook just to use the noop.
"""

from typing import Dict

from litellm.managed_agents.vms.base import AgentVMProvider
from litellm.managed_agents.vms.noop import NoopProvider

_REGISTRY: Dict[str, AgentVMProvider] = {}


def register_vm_provider(provider: AgentVMProvider) -> None:
    """Register a provider by ``provider.name``. Last-write-wins; tests
    use this to swap in a fresh ``NoopProvider`` between cases."""
    _REGISTRY[provider.name] = provider


def get_vm_provider(name: str) -> AgentVMProvider:
    """Return the registered provider for ``name``.

    Lazily instantiates a default ``NoopProvider`` on first access so tests
    don't need a setup hook just to use the noop. For other names (e.g.
    ``"ec2"``) the caller MUST register an instance first via
    ``register_vm_provider`` (typically at proxy startup from
    ``factory.build_vm_provider``).
    """
    if name not in _REGISTRY:
        if name == "noop":
            _REGISTRY[name] = NoopProvider()
        else:
            raise KeyError(f"No VM provider registered for '{name}'")
    return _REGISTRY[name]


def reset_vm_provider_registry() -> None:
    """Test helper: drop all registered providers."""
    _REGISTRY.clear()
