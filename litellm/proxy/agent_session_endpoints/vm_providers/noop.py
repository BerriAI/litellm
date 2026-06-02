"""
No-op `AgentVMProvider` implementation.

Used in tests and for environments without an EC2 backend. `provision` returns
a fake handle immediately; `status` always reports `running`. The provider is
the default in `factory.py` so the proxy starts up without AWS configured.
"""

from __future__ import annotations

import uuid

from litellm.proxy.agent_session_endpoints.vm_providers.base import (
    AgentVMProvider,
    ProvisionContext,
    VMHandle,
    VMState,
    VMStatus,
)


class NoopProvider(AgentVMProvider):
    """In-memory provider used by tests and `vm_provider: noop` config."""

    name = "noop"

    def __init__(self) -> None:
        self._terminated: set = set()

    async def provision(self, ctx: ProvisionContext) -> VMHandle:
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

    async def terminate(self, vm: VMHandle) -> None:
        self._terminated.add(vm.vm_id)

    async def status(self, vm: VMHandle) -> VMStatus:
        if vm.vm_id in self._terminated:
            return VMStatus(state=VMState.TERMINATED)
        return VMStatus(state=VMState.RUNNING, public_ip="127.0.0.1")
