"""Pluggable sandboxes: where tool calls actually execute."""

from litellm.managed_agents.sandbox.base import Sandbox, ToolResult
from litellm.managed_agents.sandbox.ec2_ssm import EC2SandboxViaSSM
from litellm.managed_agents.sandbox.local import LocalSandbox

__all__ = [
    "Sandbox",
    "ToolResult",
    "LocalSandbox",
    "EC2SandboxViaSSM",
]
