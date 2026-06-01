"""
LocalSandbox — executes tool calls in the proxy process.

For dev mode only. Multi-tenant unsafe (no isolation between sessions).
Production deploys should use ``EC2SandboxViaSSM`` instead.

The set of tools recognised here mirrors the names ``claude-agent-sdk``
uses for its built-ins (``Bash``, ``Read``, ``Write``, ``Edit``) so the
``LiteLLMAgentRuntime`` and any custom runtime can reuse them.

For ``ClaudeSDKAgentRuntime``, ``LocalSandbox.execute_tool`` is intentionally
unused: claude-agent-sdk runs its built-in tools in-process directly, and the
sandbox-routing override is deferred to a follow-up PR. The ``cwd`` property
is still consumed though so the SDK runs in the right directory.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from litellm.managed_agents.sandbox.base import Sandbox, ToolResult


class LocalSandbox(Sandbox):
    """Executes tools in the proxy process.

    ``working_dir`` defaults to a fresh temp dir per-instance. Pass an
    explicit dir to share state across runs (e.g. a cloned repo).

    ``shell_timeout_seconds`` bounds individual ``Bash`` calls — the LLM
    can request long-running commands and we don't want a runaway process
    holding up a whole run.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        shell_timeout_seconds: float = 60.0,
    ) -> None:
        self._working_dir: Optional[str] = working_dir
        self._owned_tmpdir: Optional[str] = None
        self._shell_timeout = shell_timeout_seconds

    @property
    def cwd(self) -> Optional[str]:
        return self._working_dir

    async def setup(self) -> None:
        if self._working_dir is None:
            import tempfile

            self._owned_tmpdir = tempfile.mkdtemp(prefix="litellm_managed_agent_")
            self._working_dir = self._owned_tmpdir

    async def teardown(self) -> None:
        if self._owned_tmpdir is not None and os.path.isdir(self._owned_tmpdir):
            shutil.rmtree(self._owned_tmpdir, ignore_errors=True)
            self._owned_tmpdir = None
            self._working_dir = None

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> ToolResult:
        if self._working_dir is None:
            await self.setup()
        # mypy: setup populates _working_dir
        cwd = self._working_dir or "."

        name = tool_name.lower()
        if name in {"bash", "shell", "exec"}:
            return await self._run_bash(tool_input, cwd)
        if name in {"read", "read_file"}:
            return self._run_read(tool_input, cwd)
        if name in {"write", "write_file"}:
            return self._run_write(tool_input, cwd)
        if name in {"edit", "edit_file"}:
            return self._run_edit(tool_input, cwd)
        if name == "ls":
            return self._run_ls(tool_input, cwd)
        return ToolResult(
            output=f"unknown tool: {tool_name}",
            is_error=True,
            metadata={"sandbox": "local"},
        )

    # ------------------------------------------------------------------
    # Tool implementations — kept tiny on purpose. These exist so
    # LiteLLMAgentRuntime + LocalSandbox can do useful work end-to-end
    # without depending on claude-agent-sdk's built-in tool surface.
    # ------------------------------------------------------------------

    async def _run_bash(self, tool_input: Dict[str, Any], cwd: str) -> ToolResult:
        cmd = tool_input.get("command")
        if not isinstance(cmd, str) or not cmd:
            return ToolResult(output="missing or empty 'command'", is_error=True)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._shell_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    output=f"command timed out after {self._shell_timeout}s",
                    is_error=True,
                    metadata={"timeout": self._shell_timeout},
                )
            output = (stdout or b"").decode(errors="replace")
            err = (stderr or b"").decode(errors="replace")
            if proc.returncode != 0:
                return ToolResult(
                    output=f"{output}\n{err}".strip(),
                    is_error=True,
                    metadata={"exit_code": proc.returncode},
                )
            return ToolResult(
                output=output,
                metadata={"exit_code": proc.returncode},
            )
        except Exception as exc:
            return ToolResult(output=f"bash failed: {exc}", is_error=True)

    def _resolve_path(self, tool_input: Dict[str, Any], cwd: str) -> Optional[Path]:
        path = tool_input.get("path") or tool_input.get("file_path")
        if not isinstance(path, str) or not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(cwd) / p
        return p

    def _run_read(self, tool_input: Dict[str, Any], cwd: str) -> ToolResult:
        p = self._resolve_path(tool_input, cwd)
        if p is None:
            return ToolResult(output="missing 'path'", is_error=True)
        try:
            return ToolResult(output=p.read_text())
        except FileNotFoundError:
            return ToolResult(output=f"no such file: {p}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"read failed: {exc}", is_error=True)

    def _run_write(self, tool_input: Dict[str, Any], cwd: str) -> ToolResult:
        p = self._resolve_path(tool_input, cwd)
        if p is None:
            return ToolResult(output="missing 'path'", is_error=True)
        content = tool_input.get("content", "")
        if not isinstance(content, str):
            return ToolResult(output="'content' must be a string", is_error=True)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return ToolResult(output=f"wrote {len(content)} bytes to {p}")
        except Exception as exc:
            return ToolResult(output=f"write failed: {exc}", is_error=True)

    def _run_edit(self, tool_input: Dict[str, Any], cwd: str) -> ToolResult:
        p = self._resolve_path(tool_input, cwd)
        if p is None:
            return ToolResult(output="missing 'path'", is_error=True)
        old = tool_input.get("old_string")
        new = tool_input.get("new_string", "")
        if not isinstance(old, str) or not isinstance(new, str):
            return ToolResult(
                output="'old_string' and 'new_string' must be strings", is_error=True
            )
        try:
            text = p.read_text()
            if old not in text:
                return ToolResult(
                    output="'old_string' not found in file", is_error=True
                )
            p.write_text(text.replace(old, new, 1))
            return ToolResult(output=f"edited {p}")
        except FileNotFoundError:
            return ToolResult(output=f"no such file: {p}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"edit failed: {exc}", is_error=True)

    def _run_ls(self, tool_input: Dict[str, Any], cwd: str) -> ToolResult:
        path = tool_input.get("path") or cwd
        try:
            entries = sorted(os.listdir(path))
            return ToolResult(output="\n".join(entries))
        except FileNotFoundError:
            return ToolResult(output=f"no such dir: {path}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"ls failed: {exc}", is_error=True)
