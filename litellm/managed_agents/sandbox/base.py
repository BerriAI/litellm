"""
Sandbox abstraction — the venue where tool calls actually execute.

The runtime separates *deciding what tool to call* (LLM tool loop) from
*actually running the tool* (filesystem, shell, network). This lets the
same runtime drive a tool loop against:

  * ``LocalSandbox`` — execute in the proxy process (dev only)
  * ``EC2SandboxViaSSM`` — execute on a remote VM via SSM RunCommand
  * ``DockerSandbox`` — execute in a container (future)

Each implementation only needs to honor ``execute_tool(name, input)``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result of executing a tool inside a sandbox.

    ``output`` is whatever the tool returned (string, dict, bytes — caller
    decides). ``is_error`` is True when the tool failed; the runtime maps
    this onto the ``tool_result`` event ``is_error`` field which the LLM
    then sees on its next turn. ``metadata`` is open-ended for sandbox
    implementations that want to surface execution-venue details (exit
    code, vm_id, region, duration_ms, etc.).
    """

    output: Any
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class Sandbox(ABC):
    """Where tool calls execute.

    Subclasses implement ``execute_tool(tool_name, tool_input) -> ToolResult``.

    The runtime calls into this for any tool the LLM requests. For example,
    when the LLM emits a ``tool_use`` block ``{"name": "Bash", "input":
    {"command": "ls"}}``, the runtime invokes
    ``await sandbox.execute_tool("Bash", {"command": "ls"})`` and feeds the
    result back into the next LLM turn as a ``tool_result``.

    Implementations should be safe to share across concurrent runs only if
    they document so explicitly. The default contract is "one Sandbox per
    Session" — each Session owns its sandbox for the duration of its life.
    """

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool and return its result.

        Implementations must NOT raise on tool-level errors (e.g. the
        command exited non-zero, the file did not exist). Instead, return
        ``ToolResult(output=<error msg>, is_error=True)`` so the LLM can
        see the failure and react. Raise only on infrastructure failures
        (sandbox unreachable, OOM) so the runtime can decide whether to
        abort the run or retry.
        """

    async def setup(self) -> None:
        """Optional: prepare the sandbox before the first tool call.

        Override for sandboxes that need to provision a VM, clone repos,
        install deps, etc. Default is no-op so simple sandboxes
        (``LocalSandbox``) can ignore it.
        """
        return None

    async def teardown(self) -> None:
        """Optional: clean up after the last tool call.

        Override for sandboxes that need to release a VM, delete a
        container, etc. Default is no-op.
        """
        return None

    @property
    def cwd(self) -> Optional[str]:
        """Optional working directory hint for runtimes that need one
        (e.g. claude-agent-sdk's ``ClaudeAgentOptions.cwd``).

        Returning ``None`` means "no hint" — the runtime falls back to
        whatever default it normally uses.
        """
        return None
