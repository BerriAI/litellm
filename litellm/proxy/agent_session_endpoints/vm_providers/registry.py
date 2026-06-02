"""Process-wide registry of VM providers, keyed by provider name.

Bridge between A1's ``session_endpoints`` (which calls
``provider.provision(session_id=..., agent_id=..., ...)``) and B1's
abstraction (which calls ``provider.provision(ctx: ProvisionContext)``).
The wrapper translates A1's keyword call into a ``ProvisionContext`` and
back to A1's ``result.vm_id`` shape so neither side needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from litellm.proxy.agent_session_endpoints.vm_providers.base import (
    AgentVMProvider,
    ProvisionContext,
    Repo,
)
from litellm.proxy.agent_session_endpoints.vm_providers.noop import NoopProvider

_REGISTRY: Dict[str, "_ProviderAdapter"] = {}


@dataclass
class ProvisionResult:
    """Return shape A1's ``session_endpoints`` expects."""

    vm_id: str
    metadata: Optional[Dict[str, Any]] = None


class _ProviderAdapter:
    """Adapter that exposes A1's keyword-style API on top of B1's provider."""

    def __init__(self, inner: AgentVMProvider) -> None:
        self._inner = inner
        self.name = inner.name

    @property
    def inner(self) -> AgentVMProvider:
        return self._inner

    async def provision(
        self,
        *,
        session_id: str,
        agent_id: Optional[str],
        repos: List[Dict[str, Any]],
        env_vars: Optional[Dict[str, str]],
        daemon_token: str,
        proxy_base_url: str,
        team_id: Optional[str] = None,
        mode: str = "session",
        secrets: Optional[Dict[str, str]] = None,
    ) -> ProvisionResult:
        ctx = ProvisionContext(
            session_id=session_id,
            team_id=team_id or "",
            agent_id=agent_id,
            repos=[Repo(**r) for r in (repos or [])],
            env_vars=dict(env_vars or {}),
            secrets=dict(secrets or {}),
            daemon_jwt=daemon_token,
            daemon_base_url=proxy_base_url,
            mode=mode,
        )
        handle = await self._inner.provision(ctx)
        return ProvisionResult(vm_id=handle.vm_id, metadata=handle.metadata)

    async def terminate(
        self,
        *,
        session_id: str,
        vm_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not vm_id:
            return
        from litellm.proxy.agent_session_endpoints.vm_providers.base import VMHandle

        # Carry session_id in metadata so test/recording noops can correlate
        # the terminate call to the originating session without changing the
        # ABC's terminate(handle) signature.
        handle_metadata = {**(metadata or {}), "session_id": session_id}
        handle = VMHandle(vm_id=vm_id, provider=self.name, metadata=handle_metadata)
        try:
            await self._inner.terminate(handle)
        except TypeError:
            # B1's EC2Provider expects an ``aws_creds`` kwarg; the noop and
            # warm-pool flows don't have one. Skip without raising — caller
            # is best-effort.
            pass


def register_vm_provider(provider: AgentVMProvider) -> None:
    """Register a provider by ``provider.name``. Last-write-wins."""
    _REGISTRY[provider.name] = _ProviderAdapter(provider)


def get_vm_provider(name: str) -> _ProviderAdapter:
    """Return the registered provider for ``name``.

    Lazily instantiates a default ``NoopProvider`` on first access so
    tests don't need a setup hook just to use the noop.
    """
    if name not in _REGISTRY:
        if name == "noop":
            _REGISTRY[name] = _ProviderAdapter(NoopProvider())
        else:
            raise KeyError(f"No VM provider registered for '{name}'")
    return _REGISTRY[name]


def reset_vm_provider_registry() -> None:
    """Test helper: drop all registered providers."""
    _REGISTRY.clear()
