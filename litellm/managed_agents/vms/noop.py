"""
No-op `AgentVMProvider` implementation.

Used in tests and for environments without an EC2 backend. `provision` returns
a fake handle immediately; `status` always reports `running`. The provider is
the default in `factory.py` so the proxy starts up without AWS configured.

Records every `provision`/`terminate` call so legacy A-era tests can assert
the endpoints invoked the provider with the expected arguments. Recording is
thread-safe.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

from litellm.managed_agents.vms.base import (
    AgentVMProvider,
    ProvisionContext,
    VMHandle,
    VMState,
    VMStatus,
)


class NoopProvider(AgentVMProvider):
    """In-memory provider used by tests and `vm_provider: noop` config."""

    name = "noop"

    def __init__(self, fail_provision: bool = False) -> None:
        self._lock = threading.Lock()
        self._terminated: set = set()
        # Recording for legacy A-era tests.
        self.provision_calls: List[Dict[str, Any]] = []
        self.terminate_calls: List[Dict[str, Any]] = []
        self.fail_provision = fail_provision

    async def provision(self, ctx: ProvisionContext) -> VMHandle:
        with self._lock:
            self.provision_calls.append(
                {
                    "session_id": ctx.session_id,
                    "team_id": ctx.team_id,
                    "agent_id": ctx.agent_id,
                    "repos": [
                        {"url": r.url, "ref": r.ref, "path": r.path}
                        for r in ctx.repos
                    ],
                    "env_vars_set": list((ctx.env_vars or {}).keys()),
                    "mode": ctx.mode,
                    "daemon_base_url": ctx.daemon_base_url,
                }
            )
        if self.fail_provision:
            raise RuntimeError("noop provider configured to fail provisioning")
        vm_id = f"noop-{uuid.uuid4().hex[:12]}"
        return VMHandle(
            vm_id=vm_id,
            provider=self.name,
            region=(ctx.ec2_config.region if ctx.ec2_config else None),
            metadata={
                "session_id": ctx.session_id,
                "team_id": ctx.team_id,
                "mode": ctx.mode,
            },
        )

    async def terminate(
        self,
        vm: Optional[VMHandle] = None,
        *,
        session_id: Optional[str] = None,
        vm_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Idempotent terminate.

        Accepts either B's API (``vm: VMHandle``) or the legacy keyword form
        (``session_id=..., vm_id=..., metadata=...``) used by A-era callers
        that may still construct a partial handle from a session row.
        """
        if vm is not None:
            recorded_session = (vm.metadata or {}).get("session_id")
            recorded_vm_id = vm.vm_id
            recorded_meta = vm.metadata
        else:
            recorded_session = session_id
            recorded_vm_id = vm_id
            recorded_meta = metadata

        with self._lock:
            self.terminate_calls.append(
                {
                    "session_id": recorded_session,
                    "vm_id": recorded_vm_id,
                    "metadata": recorded_meta,
                }
            )
            if recorded_vm_id:
                self._terminated.add(recorded_vm_id)

    async def status(self, vm: VMHandle) -> VMStatus:
        if vm.vm_id in self._terminated:
            return VMStatus(state=VMState.TERMINATED)
        return VMStatus(state=VMState.RUNNING, public_ip="127.0.0.1")

    def reset(self) -> None:
        """Test helper: clear recorded calls between cases."""
        with self._lock:
            self.provision_calls.clear()
            self.terminate_calls.clear()
