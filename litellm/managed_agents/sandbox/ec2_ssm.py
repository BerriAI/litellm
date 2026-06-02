"""
EC2SandboxViaSSM — placeholder.

Production sandbox: the proxy issues an SSM ``RunCommand`` to a VM
provisioned via Epic B's vm_providers (default: EC2 in customer AWS) and
shells out the tool call there. The VM no longer needs a custom binary —
stock Ubuntu + the SSM agent is enough.

This placeholder exists so:

  * The public API (``from litellm.managed_agents.sandbox import
    EC2SandboxViaSSM``) is stable from day one.
  * Validation criterion #1 (clean import) passes.
  * Future PR can swap in the real implementation without touching call
    sites.

The real implementation belongs in a follow-up that wires
``litellm/proxy/agent_session_endpoints/vm_providers/`` (already on the
integration branch) into ``execute_tool``. Tracked as a deferred item on
LIT-2879.
"""

from typing import Any, Dict, Optional

from litellm.managed_agents.sandbox.base import Sandbox, ToolResult


class EC2SandboxViaSSM(Sandbox):
    """Placeholder for SSM-backed remote execution. Not yet implemented.

    Construct it freely (the public API is stable). Calling
    ``execute_tool`` raises ``NotImplementedError`` until the SSM wiring
    lands. The constructor takes ``team_id`` because the real implementation
    will look up team-scoped AWS creds + the VM provisioning provider via
    ``vm_providers.get_vm_provider("ec2")``.
    """

    def __init__(
        self,
        team_id: Optional[str] = None,
        vm_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.team_id = team_id
        self.vm_id = vm_id
        self.region = region

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> ToolResult:
        raise NotImplementedError(
            "EC2SandboxViaSSM is a placeholder. Wire it to "
            "litellm/proxy/agent_session_endpoints/vm_providers/ec2 in a "
            "follow-up PR."
        )
