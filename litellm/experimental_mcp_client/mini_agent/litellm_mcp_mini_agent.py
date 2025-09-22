from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# --- Minimal Router call wrapper -------------------------------------------------
async def arouter_call(*, model: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Any:
    """Minimal async wrapper for a chat call.

    Tests monkeypatch this function; default implementation calls litellm.acompletion.
    """
    import litellm

    return await litellm.acompletion(model=model, messages=messages, stream=stream, **kwargs)


# --- Agent config & result types -------------------------------------------------
@dataclass
class AgentConfig:
    model: str
    max_iterations: int = 4
    max_tools_per_iter: int = 4
    enable_repair: bool = True
    research_on_unsure: bool = False
    max_research_hops: int = 1
    max_history_messages: int = 50
    hard_char_budget: int = 16000
    shell_allow_prefixes: Sequence[str] = ("echo",)
    tool_timeout_sec: float = 10.0


@dataclass
class IterationRecord:
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentRunResult:
    messages: List[Dict[str, Any]]
    final_answer: Optional[str]
    stopped_reason: str
    iterations: List[IterationRecord]


# --- MCP invoker base and locals -------------------------------------------------
class MCPInvoker:
    async def list_openai_tools(self) -> List[Dict[str, Any]]:  # OpenAI tools schema
        raise NotImplementedError

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:  # returns tool JSON string
        raise NotImplementedError


class EchoMCP(MCPInvoker):
    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo back text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        f = openai_tool.get("function", {})
        name = f.get("name")
        args = f.get("arguments")
        try:
            args_obj = json.loads(args) if isinstance(args, str) else (args or {})
        except Exception:
            args_obj = {}
        if name == "echo":
            return json.dumps({"ok": True, "name": name, "text": args_obj.get("text", ""), "result": args_obj.get("text", "")})
        return json.dumps({"ok": False, "name": name, "error": "unknown tool"})


class LocalMCPInvoker(MCPInvoker):
    def __init__(self, shell_allow_prefixes: Sequence[str] = ("echo",), tool_timeout_sec: float = 10.0) -> None:
        self.shell_allow_prefixes = tuple(shell_allow_prefixes)
        self.tool_timeout_sec = tool_timeout_sec

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "exec_python",
                    "description": "Execute short Python code and return stdout/stderr and return code.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "exec_shell",
                    "description": "Run a shell command (allowlist prefixes only).",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "research_echo",
                    "description": "Toy research tool returning a canned answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ]

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        f = openai_tool.get("function", {})
        name = f.get("name")
        args_raw = f.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}

        if name == "exec_python":
            return await self._exec_python(args.get("code", ""))
        if name == "exec_shell":
            return await self._exec_shell(args.get("cmd", ""))
        if name == "research_echo":
            q = args.get("query", "")
            return json.dumps({"ok": True, "name": name, "answer": f"echo: {q}", "citations": []})
        return json.dumps({"ok": False, "name": name, "error": "unknown tool"})

    async def _exec_python(self, code: str) -> str:
        import sys
        import tempfile
        import asyncio

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.tool_timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                return json.dumps({"ok": False, "name": "exec_python", "error": f"timeout after {self.tool_timeout_sec}s"})
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            payload = {
                "ok": rc == 0,
                "name": "exec_python",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
            }
            if rc != 0:
                tail = (stderr_s or stdout_s)[:200]
                payload["error"] = f"rc={rc}: {tail}"
            return json.dumps(payload)
        finally:
            try:
                import os
                os.unlink(path)
            except Exception:
                pass

    async def _exec_shell(self, cmd: str) -> str:
        import shlex
        import asyncio
        import contextlib

        parts = shlex.split(cmd)
        allowed = any(cmd.strip().startswith(p) for p in self.shell_allow_prefixes)
        if not allowed:
            return json.dumps({"ok": False, "name": "exec_shell", "error": "cmd prefix not allowed"})

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.tool_timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                return json.dumps({"ok": False, "name": "exec_shell", "error": f"timeout after {self.tool_timeout_sec}s"})
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            payload = {
                "ok": rc == 0,
                "name": "exec_shell",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
            }
            if rc != 0:
                tail = (stderr_s or stdout_s)[:200]
                payload["error"] = f"rc={rc}: {tail}"
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"ok": False, "name": "exec_shell", "error": str(e)})


# --- Utility helpers -------------------------------------------------------------

def _get_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """Return tool_calls list from dict or object-shaped message."""
    if not msg:
        return []
    tc = None
    if isinstance(msg, dict):
        tc = msg.get("tool_calls")
    else:
        tc = getattr(msg, "tool_calls", None)
    return tc or []


def _extract_assistant_message(resp: Any) -> Dict[str, Any]:
    """Normalize dict/object responses to a dict with role/content/tool_calls."""
    if isinstance(resp, dict):
        msg = (resp.get("choices", [{}])[0] or {}).get("message", {})
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        return {"role": role, "content": content, "tool_calls": tool_calls}
    # object shaped
    choices = getattr(resp, "choices", [])
    msg_obj = getattr(choices[0], "message", None) if choices else None
    role = getattr(msg_obj, "role", None)
    content = getattr(msg_obj, "content", None)
    tool_calls = getattr(msg_obj, "tool_calls", None) or []
    return {"role": role, "content": content, "tool_calls": tool_calls}


def _prune_history_preserve_pair(messages: List[Dict[str, Any]], max_non_system: int, hard_char_budget: int) -> List[Dict[str, Any]]:
    """Keep system messages + last tool_call pair + up to max_non_system others under a char budget."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Always try to preserve the last assistant tool_call + its tool reply, if present
    last_tool_idx = None
    last_tool_call_id = None
    for i in range(len(non_system) - 1, -1, -1):
        m = non_system[i]
        if m.get("role") == "assistant" and _get_tool_calls(m):
            last_tool_idx = i
            last_tool_call_id = _get_tool_calls(m)[-1].get("id")
            break

    # Build a set of indices to keep, preserving chronological order
    keep_idx = set()
    if last_tool_idx is not None:
        keep_idx.add(last_tool_idx)
        for j in range(last_tool_idx + 1, len(non_system)):
            if non_system[j].get("role") == "tool" and non_system[j].get("tool_call_id") == last_tool_call_id:
                keep_idx.add(j)
                break

    # Add up to max_non_system most recent non-system indices
    recent = list(range(max(0, len(non_system) - max_non_system), len(non_system)))
    for i in recent:
        keep_idx.add(i)

    # Emit in original order: system first, then chosen non-system in order
    trimmed_non_system = [non_system[i] for i in range(len(non_system)) if i in keep_idx]
    out = system_msgs + trimmed_non_system
    # Enforce rough character budget by removing oldest non-system if needed
    def total_chars(ms: List[Dict[str, Any]]) -> int:
        s = 0
        for m in ms:
            c = m.get("content")
            if isinstance(c, str):
                s += len(c)
        return s

    while total_chars(out) > hard_char_budget and any(m.get("role") != "system" for m in out):
        # drop the oldest non-system message
        for i, m in enumerate(out):
            if m.get("role") != "system":
                out.pop(i)
                break
    return out


# --- The agent loop --------------------------------------------------------------
async def arun_mcp_mini_agent(
    messages: List[Dict[str, Any]],
    *,
    mcp: MCPInvoker,
    cfg: AgentConfig,
) -> AgentRunResult:
    conv = list(messages)
    iters: List[IterationRecord] = []

    tools = await mcp.list_openai_tools()
    for step in range(cfg.max_iterations):
        # pruning for bounded context
        conv = _prune_history_preserve_pair(conv, max_non_system=min(cfg.max_history_messages, 50), hard_char_budget=cfg.hard_char_budget)

        resp = await arouter_call(model=cfg.model, messages=conv, tools=tools, tool_choice="auto")
        asst = _extract_assistant_message(resp)
        conv.append({k: v for k, v in asst.items() if k in ("role", "content", "tool_calls")})

        tcs = _get_tool_calls(asst)
        if not tcs:
            # possible final answer
            content = asst.get("content") or ""
            # research nudge if unsure and allowed
            if cfg.research_on_unsure and "not sure" in content.lower():
                conv.append({
                    "role": "assistant",
                    "content": "Observation: Model is unsure. Use available research tools and try again with citations.",
                })
                continue
            return AgentRunResult(messages=conv, final_answer=content, stopped_reason="success", iterations=iters)

        # Handle tool calls
        inv_recs: List[Dict[str, Any]] = []
        for i, tc in enumerate(tcs):
            if i >= cfg.max_tools_per_iter:
                break
            tool_name = (tc.get("function") or {}).get("name")
            args = (tc.get("function") or {}).get("arguments")
            try:
                out_json = await mcp.call_openai_tool({"id": tc.get("id"), "type": tc.get("type"), "function": {"name": tool_name, "arguments": args}})
                inv = json.loads(out_json)
                ok = bool(inv.get("ok"))
                text = inv.get("result") or inv.get("answer") or inv.get("text") or ""
                conv.append({"role": "tool", "tool_call_id": tc.get("id"), "content": text})
                inv_recs.append({"name": tool_name, **inv})
            except Exception as e:
                err = str(e)
                conv.append({"role": "tool", "tool_call_id": tc.get("id"), "content": f"error: {err}"})
                inv_recs.append({"name": tool_name, "ok": False, "error": err})

        iters.append(IterationRecord(tool_invocations=inv_recs))

        # If repair is enabled, append a compact observation for next step
        if cfg.enable_repair and inv_recs:
            # Construct a small observation message
            preview = json.dumps({"invocations": inv_recs})
            preview = (preview[:1000] + "...") if len(preview) > 1000 else preview
            directive = "If errors are present above, fix the approach and try again."
            conv.append({"role": "assistant", "content": f"Observation from last tool run (preview):\n{preview}\n\n{directive}"})

    return AgentRunResult(messages=conv, final_answer=None, stopped_reason="max_iterations", iterations=iters)


def run_mcp_mini_agent(
    messages: List[Dict[str, Any]],
    *,
    mcp: MCPInvoker,
    cfg: AgentConfig,
) -> AgentRunResult:
    return asyncio.get_event_loop().run_until_complete(arun_mcp_mini_agent(messages, mcp=mcp, cfg=cfg))
