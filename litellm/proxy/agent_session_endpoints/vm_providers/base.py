"""Abstract base class for agent session VM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProvisionResult:
    """Result returned by ``provision``.

    `vm_id` is the provider-native identifier (e.g. EC2 instance id).
    `metadata` is opaque provider state stored on the session row for
    later termination.
    """

    vm_id: str
    metadata: Optional[Dict[str, Any]] = None


class AgentVMProvider(ABC):
    """Pluggable VM backend for agent sessions.

    Concrete implementations:
      * ``NoopVMProvider``       — for tests; returns canned ids
      * ``EC2VMProvider``        — Epic B
      * ``VercelSandboxProvider``— future
    """

    name: str = "base"

    @abstractmethod
    async def provision(
        self,
        session_id: str,
        agent_id: str,
        repos: List[Dict[str, Any]],
        env_vars: Optional[Dict[str, str]],
        daemon_token: str,
        proxy_base_url: str,
    ) -> ProvisionResult:
        """Provision a new VM for ``session_id``.

        Implementations MUST be idempotent — they may be called twice if
        the cleanup sweeper retries a stuck-provisioning session.
        """

    @abstractmethod
    async def terminate(
        self,
        session_id: str,
        vm_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Tear down a VM. MUST be idempotent — terminating an already-gone
        VM is a no-op."""
