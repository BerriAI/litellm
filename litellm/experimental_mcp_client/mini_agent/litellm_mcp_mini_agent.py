"""Core mini-agent loop plus Router helpers used by debug tooling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# --- Minimal Router call wrapper -------------------------------------------------
async def arouter_call(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    stream: bool = False,
    provider_resolver=None,
    api_base=None,
    api_key=None,
    **kwargs,
) -> Any:
    """
    Unified router entrypoint for chat calls.

    - Tests can pass a provider_resolver: callable(model, api_base, api_key) -> (model, provider, api_base, api_key)
      to force an explicit provider/api_base/key without relying on auto-detection.
    """
    import litellm

    # Optional resolver seam for tests to enforce explicit provider/api settings
    if callable(provider_resolver):
        try:
            _res = provider_resolver(model, api_base, api_key)
            _m = _prov = _base = _key = None
            if isinstance(_res, (list, tuple)) and len(_res) == 4:
                _m, _prov, _base, _key = _res
            elif isinstance(_res, dict):
                _m = _res.get("model")
                _prov = _res.get("provider") or _res.get("custom_llm_provider")
                _base = _res.get("api_base")
                _key = _res.get("api_key")
            # apply if present (allow explicit None to mean 'no change')
            if _m:
                model = _m
            if _prov:
                kwargs["custom_llm_provider"] = _prov
            if _base is not None:
                api_base = _base
            if _key is not None:
                api_key = _key
        except Exception:
            # Fall through to default behavior if resolver misbehaves
            pass

    if api_base is not None:
        kwargs["api_base"] = api_base
    if api_key is not None:
        kwargs["api_key"] = api_key

    # Dev/test shortcut: if model is 'dummy' or 'noop', bypass provider mapping/network.
    # Controlled by MINI_AGENT_ALLOW_DUMMY (opt-in with "1" to keep local dev/compose healthchecks green).
    if (model in ("dummy", "noop")) and (os.getenv("MINI_AGENT_ALLOW_DUMMY", "0") == "1"):
        return {"choices": [{"message": {"role": "assistant", "content": "Dummy mini-agent response"}}]}

    # Fork-local mapping: support "chutes/<vendor>/<model>" without adding a global provider.
    # Translate to OpenAI-compatible call using CHUTES_* env when present.
    if isinstance(model, str) and model.startswith("chutes/"):
        try:
            remainder = model.split("/", 1)[1] if "/" in model else model
        except Exception:
            remainder = model
        ch_api_base = os.getenv("CHUTES_API_BASE") or os.getenv("CHUTES_BASE")
        if not ch_api_base:
            # Sensible default for Chutes OpenAI-compatible endpoint
            ch_api_base = "https://llm.chutes.ai/v1"
        ch_api_key = os.getenv("CHUTES_API_KEY") or os.getenv("CHUTES_API_TOKEN")
        if ch_api_base and not kwargs.get("api_base"):
            kwargs["api_base"] = ch_api_base
        if ch_api_key and not kwargs.get("api_key"):
            kwargs["api_key"] = ch_api_key
        kwargs.setdefault("custom_llm_provider", "openai")
        model = remainder

    return await litellm.acompletion(model=model, messages=messages, stream=stream, **kwargs)


# --- Agent config & result types -------------------------------------------------
@dataclass
class AgentConfig:
    model: str
    max_iterations: int = 3
    max_tools_per_iter: int = 4
    tool_concurrency: int = 1
    enable_repair: bool = True
    research_on_unsure: bool = False
    max_research_hops: int = 1
    max_history_messages: int = 50
    hard_char_budget: int = 16000
    shell_allow_prefixes: Sequence[str] = ("echo",)
    tool_timeout_sec: float = 10.0
    max_total_seconds: Optional[float] = None

    # Compatibility toggles surfaced to deterministic harnesses
    use_tools: bool = False
    auto_run_code_on_code_block: bool = False
    escalate_on_budget_exceeded: bool = False
    escalate_model: Optional[str] = None
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)
    router_call_ms: float = 0.0


@dataclass
class AgentRunResult:
    messages: List[Dict[str, Any]]
    final_answer: Optional[str]
    stopped_reason: str
    iterations: List[IterationRecord]
    metrics: Dict[str, Any] = field(default_factory=dict)
    used_model: Optional[str] = None


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
            {
                "type": "function",
                "function": {
                    "name": "compress_runs",
                    "description": "Compress runs of repeated characters; helper for deterministic readiness runs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"s": {"type": "string"}},
                        "required": ["s"],
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

        normalized = (name or "").strip().lower()
        if normalized in {"exec_python", "exec_code", "__main__"}:
            code = args.get("code") or args.get("source") or ""
            return await self._exec_python(code)
        if normalized == "exec_shell":
            return await self._exec_shell(args.get("cmd", ""))
        if normalized == "research_echo":
            q = args.get("query", "")
            return json.dumps({"ok": True, "name": name, "answer": f"echo: {q}", "citations": []})
        if normalized == "compress_runs":
            s = args.get("s", "")
            if not isinstance(s, str) or not s:
                return json.dumps({"ok": False, "name": name, "error": "missing string 's'"})
            out = []
            count = 1
            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    count += 1
                else:
                    out.append(f"{s[i-1]}{count}")
                    count = 1
            if s:
                out.append(f"{s[-1]}{count}")
            return json.dumps({"ok": True, "name": name, "result": "".join(out)})
        return json.dumps({"ok": False, "name": name, "error": "unknown tool"})

    async def _exec_python(self, code: str) -> str:
        import sys
        import tempfile
        import asyncio
        import time

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            t0 = time.perf_counter()
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
                t_ms = (time.perf_counter() - t0) * 1000.0
                return json.dumps({"ok": False, "name": "exec_python", "error": f"timeout after {self.tool_timeout_sec}s", "t_ms": t_ms})
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            t_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "ok": rc == 0,
                "name": "exec_python",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "t_ms": t_ms,
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
        import time

        parts = shlex.split(cmd)
        allowed = any(cmd.strip().startswith(p) for p in self.shell_allow_prefixes)
        if not allowed:
            return json.dumps({"ok": False, "name": "exec_shell", "error": "cmd prefix not allowed"})

        try:
            t0 = time.perf_counter()
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
                t_ms = (time.perf_counter() - t0) * 1000.0
                return json.dumps({"ok": False, "name": "exec_shell", "error": f"timeout after {self.tool_timeout_sec}s", "t_ms": t_ms})
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            t_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "ok": rc == 0,
                "name": "exec_shell",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "t_ms": t_ms,
            }
            if rc != 0:
                tail = (stderr_s or stdout_s)[:200]
                payload["error"] = f"rc={rc}: {tail}"
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"ok": False, "name": "exec_shell", "error": str(e)})


class DockerMCPInvoker(MCPInvoker):
    """Execute tools inside a Docker container via docker exec, mirroring LocalMCPInvoker safety."""

    def __init__(
        self,
        *,
        container: str,
        shell_allow_prefixes: Sequence[str] = ("echo",),
        tool_timeout_sec: float = 10.0,
        docker_exec_bin: str = "docker",
        python_bin: str = "python",
        shell_bin: str = "/bin/sh",
    ) -> None:
        if not container or not isinstance(container, str):
            raise ValueError("DockerMCPInvoker requires a non-empty 'container' name/id")
        self.container = container
        self.shell_allow_prefixes = tuple(shell_allow_prefixes)
        self.tool_timeout_sec = tool_timeout_sec
        self.docker_exec_bin = docker_exec_bin
        self.python_bin = python_bin
        self.shell_bin = shell_bin

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "exec_python",
                    "description": "Execute short Python code (inside container).",
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
                    "description": "Run a shell command (inside container, allowlist prefixes only).",
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
            {
                "type": "function",
                "function": {
                    "name": "compress_runs",
                    "description": "Compress runs of repeated characters.",
                    "parameters": {
                        "type": "object",
                        "properties": {"s": {"type": "string"}},
                        "required": ["s"],
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
        normalized = (name or "").strip().lower()
        if normalized in {"exec_python", "exec_code", "__main__"}:
            code = args.get("code") or args.get("source") or ""
            return await self._exec_python_docker(code)
        if normalized == "exec_shell":
            return await self._exec_shell_docker(args.get("cmd", ""))
        if normalized == "research_echo":
            q = args.get("query", "")
            return json.dumps({"ok": True, "name": name, "answer": f"echo: {q}", "citations": []})
        if normalized == "compress_runs":
            s = args.get("s", "")
            if not isinstance(s, str) or not s:
                return json.dumps({"ok": False, "name": name, "error": "missing string 's'"})
            out, count = [], 1
            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    count += 1
                else:
                    out.append(f"{s[i-1]}{count}")
                    count = 1
            if s:
                out.append(f"{s[-1]}{count}")
            return json.dumps({"ok": True, "name": name, "result": "".join(out)})
        return json.dumps({"ok": False, "name": name, "error": "unknown tool"})

    async def _exec_python_docker(self, code: str) -> str:
        import asyncio, time, contextlib

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_exec_bin,
                "exec",
                "-i",
                self.container,
                self.python_bin,
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(input=(code or "").encode("utf-8")),
                    timeout=self.tool_timeout_sec,
                )
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                t_ms = (time.perf_counter() - t0) * 1000.0
                return json.dumps(
                    {
                        "ok": False,
                        "name": "exec_python",
                        "error": f"timeout after {self.tool_timeout_sec}s",
                        "t_ms": t_ms,
                        "docker": True,
                        "container": self.container,
                    }
                )
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            t_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "ok": rc == 0,
                "name": "exec_python",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "t_ms": t_ms,
                "docker": True,
                "container": self.container,
            }
            if rc != 0:
                payload["error"] = f"rc={rc}: {(stderr_s or stdout_s)[:200]}"
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"ok": False, "name": "exec_python", "error": str(e), "docker": True, "container": self.container})

    async def _exec_shell_docker(self, cmd: str) -> str:
        import asyncio, time, contextlib

        if not any((cmd or "").strip().startswith(p) for p in self.shell_allow_prefixes):
            return json.dumps(
                {
                    "ok": False,
                    "name": "exec_shell",
                    "error": "cmd prefix not allowed (docker)",
                    "docker": True,
                    "container": self.container,
                }
            )

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_exec_bin,
                "exec",
                self.container,
                self.shell_bin,
                "-lc",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.tool_timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                t_ms = (time.perf_counter() - t0) * 1000.0
                return json.dumps(
                    {
                        "ok": False,
                        "name": "exec_shell",
                        "error": f"timeout after {self.tool_timeout_sec}s",
                        "t_ms": t_ms,
                        "docker": True,
                        "container": self.container,
                    }
                )
            rc = proc.returncode or 0
            stdout_s = (out or b"").decode("utf-8", errors="replace")
            stderr_s = (err or b"").decode("utf-8", errors="replace")
            t_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "ok": rc == 0,
                "name": "exec_shell",
                "rc": rc,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "t_ms": t_ms,
                "docker": True,
                "container": self.container,
            }
            if rc != 0:
                payload["error"] = f"rc={rc}: {(stderr_s or stdout_s)[:200]}"
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"ok": False, "name": "exec_shell", "error": str(e), "docker": True, "container": self.container})


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


def _safe_tool_stub(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Create a minimal assistant message preserving tool_calls."""
    tc = _get_tool_calls(msg)
    stub = {"role": "assistant", "content": ""}
    if tc:
        stub["tool_calls"] = tc
    # Preserve optional identifiers if present
    for key in ("id", "name"):
        if key in msg:
            stub[key] = msg[key]
    return stub


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


def _extract_first_code_block(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract the first fenced code block from text.
    Supports ```python ...``` or plain ``` ... ``` fences.
    Returns (language, code). Either may be None if not found.
    """
    if not text or not isinstance(text, str):
        return (None, None)
    start = None
    lang: Optional[str] = None
    # simple state machine to avoid pulling in regex for determinism
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if start is None:
            if line.startswith("```"):
                fence = line[3:].strip()
                lang = fence or None
                start = i + 1
        else:
            if line.strip().startswith("```"):
                # capture block
                code_lines = lines[start:i]
                code = "\n".join(code_lines).strip()
                return (lang, code if code else None)
        i += 1
    return (None, None)


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
    keep_idx: set[int] = set()
    essential_idx: set[int] = set()
    if last_tool_idx is not None:
        keep_idx.add(last_tool_idx)
        essential_idx.add(last_tool_idx)
        for j in range(last_tool_idx + 1, len(non_system)):
            if non_system[j].get("role") == "tool" and non_system[j].get("tool_call_id") == last_tool_call_id:
                keep_idx.add(j)
                essential_idx.add(j)
                break

    # Add up to max_non_system most recent non-system indices
    recent = list(range(max(0, len(non_system) - max_non_system), len(non_system)))
    for i in recent:
        keep_idx.add(i)

    # Emit in original order: system first, then chosen non-system in order
    trimmed_non_system = [non_system[i] for i in range(len(non_system)) if i in keep_idx]
    essential_messages = {id(non_system[i]) for i in essential_idx}
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
        removed = False
        for i, m in enumerate(out):
            if m.get("role") == "system":
                continue
            if id(m) in essential_messages:
                continue
            out.pop(i)
            removed = True
            break
        if not removed:
            # Nothing removable without breaking essential tool-call pair; stop trimming
            break

    # If essential assistant/tool messages were dropped earlier (e.g., due to copying), ensure at least one assistant tool-call remains
    has_assistant_tool = any((m.get("role") == "assistant") and _get_tool_calls(m) for m in out)
    if not has_assistant_tool and last_tool_idx is not None:
        assistant_msg = _safe_tool_stub(non_system[last_tool_idx])
        out.append(assistant_msg)

    # Ensure matching tool reply exists if we have assistant tool-call reference
    if any((m.get("role") == "assistant") and _get_tool_calls(m) for m in out):
        if last_tool_call_id is not None and not any(
            (m.get("role") == "tool" and m.get("tool_call_id") == last_tool_call_id) for m in out
        ):
            for j in range(last_tool_idx + 1, len(non_system)) if last_tool_idx is not None else []:
                tool_msg = non_system[j]
                if tool_msg.get("role") == "tool" and tool_msg.get("tool_call_id") == last_tool_call_id:
                    out.append(tool_msg)
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
    t_start = time.perf_counter()

    tools = await mcp.list_openai_tools()
    tools_to_pass = tools if cfg.use_tools else []
    tool_choice   = "auto" if cfg.use_tools else None
    metrics: Dict[str, Any] = {"escalated": False}
    used_model = cfg.model
    escalate_next = False
    completion_kwargs = dict(cfg.completion_kwargs or {})
    forced_tool_choice = completion_kwargs.pop("tool_choice", None)

    for step in range(cfg.max_iterations):
        # wall-clock budget check
        if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
            metrics["used_model"] = used_model
            return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters, metrics=metrics, used_model=used_model)

        # pruning for bounded context
        conv = _prune_history_preserve_pair(
            conv,
            max_non_system=min(cfg.max_history_messages, 50),
            hard_char_budget=cfg.hard_char_budget,
        )

        # Choose model; on final iteration optionally escalate if enabled
        chosen_model = cfg.model
        escalate_this_step = False
        if cfg.escalate_on_budget_exceeded and escalate_next:
            chosen_model = cfg.escalate_model or cfg.model
            escalate_this_step = bool(cfg.escalate_model)

        t_router0 = time.perf_counter()
        call_kwargs = dict(completion_kwargs)
        effective_tool_choice = forced_tool_choice or tool_choice
        resp = await arouter_call(
            model=chosen_model,
            messages=conv,
            tools=tools_to_pass,
            tool_choice=effective_tool_choice,
            **call_kwargs,
        )
        router_ms = (time.perf_counter() - t_router0) * 1000.0
        asst = _extract_assistant_message(resp)
        conv.append({k: v for k, v in asst.items() if k in ("role", "content", "tool_calls")})
        used_model = chosen_model
        if escalate_this_step and cfg.escalate_model:
            metrics["escalated"] = True
        escalate_next = False

        # Wall-clock cutoff after model response
        if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
            metrics["used_model"] = used_model
            return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters, metrics=metrics, used_model=used_model)
        
        tcs = _get_tool_calls(asst)
        if tcs:
            allowed_tools = {"exec_python", "exec_shell", "research_echo", "exec_code", "__main__", "python", "compress_runs", "echo"}
            filtered_calls: List[Dict[str, Any]] = []
            for tc in tcs:
                fn_name = ((tc.get("function") or {}).get("name") or "").strip().lower()
                if fn_name in allowed_tools:
                    filtered_calls.append(tc)
                else:
                    fn_payload = tc.get("function") or {}
                    args_raw = fn_payload.get("arguments")
                    args_obj = None
                    if isinstance(args_raw, str):
                        try:
                            args_obj = json.loads(args_raw)
                        except Exception:
                            args_obj = None
                    elif isinstance(args_raw, dict):
                        args_obj = args_raw

                    if fn_name in {"compress_runs", "code", "python_code"}:
                        fn_payload["name"] = "exec_python"
                        filtered_calls.append(tc)
                    elif isinstance(args_obj, dict) and "code" in args_obj:
                        fn_payload["name"] = "exec_python"
                        try:
                            fn_payload["arguments"] = json.dumps({"code": args_obj["code"]})
                        except Exception:
                            fn_payload["arguments"] = args_raw
                        filtered_calls.append(tc)
                    else:
                        conv.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "content": f"error: unsupported tool '{fn_name}'",
                            }
                        )
            tcs = filtered_calls

        # Optional: autorun code block from assistant content if no tool calls are present
        if not tcs and cfg.auto_run_code_on_code_block:
            content_for_code = asst.get("content") or ""
            lang, code = _extract_first_code_block(content_for_code)
            if code:
                try:
                    args_json = json.dumps({"code": code})
                except Exception:
                    args_json = '{"code": ""}'
                # Choose tool based on language hint: python → exec_python; anything else → exec_code
                lang_l = (lang or "").strip().lower()
                tool_name = "exec_python" if lang_l in ("", "py", "python") else "exec_code"
                tcs = [
                    {
                        "id": "auto_codeblock_0",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_json},
                    }
                ]

        if not tcs:
            content = asst.get("content")
            # JSON-mode: allow dict content to pass through without coercion
            if isinstance(content, dict):
                iters.append(IterationRecord(tool_invocations=[], router_call_ms=router_ms))
                metrics["used_model"] = used_model
                return AgentRunResult(messages=conv, final_answer=None, stopped_reason="success", iterations=iters, metrics=metrics, used_model=used_model)
            # String-mode
            content_str = (content or "") if isinstance(content, str) else ""
            if not (content_str.strip()):
                conv.append({
                    "role": "user",
                    "content": "I did not receive runnable Python. Please provide Python code in a fenced block and try again."
                })
                continue
            # If wall-clock exceeded, stop with budget instead of finalizing
            if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
                # record router timing for observability
                iters.append(IterationRecord(tool_invocations=[], router_call_ms=router_ms))
                metrics["used_model"] = used_model
                return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters, metrics=metrics, used_model=used_model)
            # NEW: if we are supposed to escalate and the model says budget exceeded, don't finalize yet
            if (
                cfg.escalate_on_budget_exceeded
                and isinstance(content, str)
                and "budget" in content.lower()
                and "exceeded" in content.lower()
            ):
                conv.append({
                    "role": "assistant",
                    "content": "Observation: budget exceeded; escalating on next step."
                })
                escalate_next = True
                continue

            if cfg.research_on_unsure and "not sure" in content.lower():
                conv.append({
                    "role": "assistant",
                    "content": "Observation: Model is unsure. Use available research tools and try again with citations.",
                })
                continue

            iters.append(IterationRecord(tool_invocations=[], router_call_ms=router_ms))
            metrics["used_model"] = used_model
            return AgentRunResult(messages=conv, final_answer=content_str, stopped_reason="success", iterations=iters, metrics=metrics, used_model=used_model)


        # Handle tool calls
        inv_recs: List[Dict[str, Any]] = []
        to_run = tcs[: cfg.max_tools_per_iter]
        tool_outputs: List[str] = []

        async def _invoke(idx: int, tc: Dict[str, Any]):
            tool_name = (tc.get("function") or {}).get("name")
            args = (tc.get("function") or {}).get("arguments")
            payload = {"id": tc.get("id"), "type": tc.get("type"), "function": {"name": tool_name, "arguments": args}}
            try:
                t0 = time.perf_counter()
                out_json = await mcp.call_openai_tool(payload)
                t_ms = (time.perf_counter() - t0) * 1000.0
                return (idx, out_json, None, tool_name, tc.get("id"), t_ms, args)
            except Exception as e:
                t_ms = (time.perf_counter() - t0) * 1000.0 if 't0' in locals() else 0.0
                return (idx, None, e, tool_name, tc.get("id"), t_ms, args)

        results = []
        if getattr(cfg, "tool_concurrency", 1) > 1 and len(to_run) > 1:
            sem = asyncio.Semaphore(max(1, int(getattr(cfg, "tool_concurrency", 1))))
            async def _runner(idx: int, tc: Dict[str, Any]):
                async with sem:
                    return await _invoke(idx, tc)
            tasks = [_runner(i, tc) for i, tc in enumerate(to_run)]
            results = await asyncio.gather(*tasks)
        else:
            for i, tc in enumerate(to_run):
                results.append(await _invoke(i, tc))

        # Append tool results in submission order
        for idx, out_json, err, tool_name, tool_call_id, t_ms, raw_args in sorted(results, key=lambda x: x[0]):
            arguments_raw: Optional[str] = None
            arguments_parsed: Optional[Dict[str, Any]] = None
            if isinstance(raw_args, str):
                arguments_raw = raw_args
                try:
                    arguments_parsed = json.loads(raw_args)
                except Exception:
                    arguments_parsed = None
            elif isinstance(raw_args, dict):
                arguments_parsed = raw_args
                try:
                    arguments_raw = json.dumps(raw_args)
                except Exception:
                    arguments_raw = None

            record: Dict[str, Any] = {
                "call_index": idx,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments_raw": arguments_raw,
                "arguments": arguments_parsed,
                "t_ms": t_ms,
                "model": chosen_model,
                "name": tool_name,  # backwards compatibility for older callers
            }

            if err is not None:
                err_s = str(err)
                conv.append({"role": "tool", "tool_call_id": tool_call_id, "content": f"error: {err_s}"})
                record.update({
                    "ok": False,
                    "rc": None,
                    "result": None,
                    "stdout": None,
                    "stderr": None,
                    "error": err_s,
                })
                inv_recs.append(record)
                continue

            try:
                inv = json.loads(out_json) if isinstance(out_json, str) else (out_json or {})
            except Exception:
                inv = {"ok": False, "error": "invalid tool JSON"}

            text = inv.get("result") or inv.get("answer") or inv.get("text") or inv.get("stdout") or ""
            conv.append({"role": "tool", "tool_call_id": tool_call_id, "content": text})

            ok_val = inv.get("ok")
            if ok_val is None:
                ok_val = True
            record.update({
                "ok": ok_val,
                "rc": inv.get("rc") or inv.get("returncode"),
                "result": inv.get("result") or inv.get("answer"),
                "stdout": inv.get("stdout"),
                "stderr": inv.get("stderr"),
                "error": inv.get("error"),
                "model": inv.get("model") or inv.get("used_model") or chosen_model,
            })

            if text:
                tool_outputs.append(text)

            inv_recs.append(record)

        iters.append(IterationRecord(tool_invocations=inv_recs, router_call_ms=router_ms))

        # If repair is enabled, append a compact observation for next step
        all_ok = all(inv.get("ok") for inv in inv_recs if isinstance(inv, dict) and "ok" in inv)
        if tool_outputs and all_ok:
            final_text = "\n".join(tool_outputs)
            metrics["used_model"] = used_model
            return AgentRunResult(messages=conv, final_answer=final_text, stopped_reason="success", iterations=iters, metrics=metrics, used_model=used_model)

        if cfg.enable_repair and inv_recs:
            preview = json.dumps({"invocations": inv_recs})
            preview = (preview[:1000] + "...") if len(preview) > 1000 else preview
            directive = "If errors are present above, fix the approach and try again."
            conv.append({
                "role": "assistant",
                "content": f"Observation from last tool run (preview):\n{preview}\n\n{directive}"
            })
            continue

    metrics["used_model"] = used_model
    return AgentRunResult(
        messages=conv,
        final_answer=None,
        stopped_reason="max_iterations",
        iterations=iters,
        metrics=metrics,
        used_model=used_model,
    )


def run_mcp_mini_agent(
    messages: List[Dict[str, Any]],
    *,
    mcp: MCPInvoker,
    cfg: AgentConfig,
) -> AgentRunResult:
    return asyncio.get_event_loop().run_until_complete(arun_mcp_mini_agent(messages, mcp=mcp, cfg=cfg))
