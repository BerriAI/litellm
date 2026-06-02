"""
Noop VM provider — for tests.

Records every call so tests can assert provider.provision/terminate were
invoked with expected arguments.
"""

import threading
import uuid
from typing import Any, Dict, List, Optional

from litellm.proxy.agent_session_endpoints.vm_providers.base import (
    AgentVMProvider,
    ProvisionResult,
)


class NoopVMProvider(AgentVMProvider):
    """In-process provider: returns ``noop_<uuid>`` instance ids.

    Thread-safe call recording. Tests inspect ``provision_calls`` /
    ``terminate_calls`` to verify the endpoints invoked the provider
    correctly.
    """

    name = "noop"

    def __init__(self, fail_provision: bool = False) -> None:
        self._lock = threading.Lock()
        self.provision_calls: List[Dict[str, Any]] = []
        self.terminate_calls: List[Dict[str, Any]] = []
        self.fail_provision = fail_provision

    async def provision(
        self,
        session_id: str,
        agent_id: str,
        repos: List[Dict[str, Any]],
        env_vars: Optional[Dict[str, str]],
        daemon_token: str,
        proxy_base_url: str,
    ) -> ProvisionResult:
        with self._lock:
            self.provision_calls.append(
                {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "repos": repos,
                    "env_vars_set": list((env_vars or {}).keys()),
                    "proxy_base_url": proxy_base_url,
                }
            )
        if self.fail_provision:
            raise RuntimeError("noop provider configured to fail provisioning")
        return ProvisionResult(
            vm_id=f"noop_{uuid.uuid4().hex[:12]}",
            metadata={"provider": "noop"},
        )

    async def terminate(
        self,
        session_id: str,
        vm_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        with self._lock:
            self.terminate_calls.append(
                {
                    "session_id": session_id,
                    "vm_id": vm_id,
                    "metadata": metadata,
                }
            )

    def reset(self) -> None:
        """Clear recorded calls — for use between tests."""
        with self._lock:
            self.provision_calls.clear()
            self.terminate_calls.clear()
