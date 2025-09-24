from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Py3.12+ safety: ensure a default event loop exists for tests that call
# asyncio.get_event_loop().run_until_complete(...) without prior loop setup.
# Belt-and-suspenders: try running + current loop checks.
try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        # No running loop; ensure a loop is set on this thread
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    except Exception:
        # Best-effort; if the test runner or framework manages the loop, avoid hard failures
        pass

# If get_event_loop() still raises on 3.12+, create and set one explicitly.
try:
    asyncio.get_event_loop()
except RuntimeError:
    try:
        _loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop2)
    except Exception:
        pass


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
    # Controlled by MINI_AGENT_ALLOW_DUMMY (default "1" keeps local dev/compose healthchecks green).
    if (model in ("dummy", "noop")) and (os.getenv("MINI_AGENT_ALLOW_DUMMY", "1") == "1"):
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    # Fork-local mapping: support "chutes/<vendor>/<model>" without adding a global provider.
    # Translate to OpenAI-compatible call using CHUTES_* env when present.
    if isinstance(model, str) and model.startswith("chutes/"):
        try:
            remainder = model.split("/", 1)[1] if "/" in model else model
        except Exception:
            remainder = model
        ch_api_base = os.getenv("CHUTES_API_BASE") or os.getenv("CHUTES_BASE")
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

    # Compatibility / optional behavior toggles used by smokes
    use_tools: bool = False  # when True, agent prefers tool-enabled prompting (kept for compatibility)
    auto_run_code_on_code_block: bool = False  # reserved; no-op in minimal agent
    escalate_on_budget_exceeded: bool = False  # if True, escalate on last iteration
    escalate_model: Optional[str] = None  # model to use when escalating (falls back to model)


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
    t_start = time.perf_counter()

    tools = await mcp.list_openai_tools()
    tools_to_pass = tools if cfg.use_tools else []
    tool_choice   = "auto" if cfg.use_tools else None

    for step in range(cfg.max_iterations):
        # wall-clock budget check
        if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
            return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters)

        # pruning for bounded context
        conv = _prune_history_preserve_pair(
            conv,
            max_non_system=min(cfg.max_history_messages, 50),
            hard_char_budget=cfg.hard_char_budget,
        )

        # Choose model; on final iteration optionally escalate if enabled
        chosen_model = cfg.model
        is_final_iter = (step == cfg.max_iterations - 1)
        if cfg.escalate_on_budget_exceeded and is_final_iter and (cfg.escalate_model or cfg.model):
            chosen_model = (cfg.escalate_model or cfg.model)

        resp = await arouter_call(
            model=chosen_model,
            messages=conv, 
            tools=tools_to_pass, 
            tool_choice=tool_choice
        )
        asst = _extract_assistant_message(resp)
        conv.append({k: v for k, v in asst.items() if k in ("role", "content", "tool_calls")})
        
        # Wall-clock cutoff after model response
        if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
            return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters)
        
        tcs = _get_tool_calls(asst)

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
            content = (asst.get("content") or "")
            # If wall-clock exceeded, stop with budget instead of finalizing
            if (cfg.max_total_seconds is not None) and ((time.perf_counter() - t_start) >= cfg.max_total_seconds):
                return AgentRunResult(messages=conv, final_answer=None, stopped_reason="budget", iterations=iters)
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
                continue

            if cfg.research_on_unsure and "not sure" in content.lower():
                conv.append({
                    "role": "assistant",
                    "content": "Observation: Model is unsure. Use available research tools and try again with citations.",
                })
                continue

            return AgentRunResult(messages=conv, final_answer=content, stopped_reason="success", iterations=iters)


        # Handle tool calls
        inv_recs: List[Dict[str, Any]] = []
        to_run = tcs[: cfg.max_tools_per_iter]

        async def _invoke(idx: int, tc: Dict[str, Any]):
            tool_name = (tc.get("function") or {}).get("name")
            args = (tc.get("function") or {}).get("arguments")
            payload = {"id": tc.get("id"), "type": tc.get("type"), "function": {"name": tool_name, "arguments": args}}
            try:
                out_json = await mcp.call_openai_tool(payload)
                return (idx, out_json, None, tool_name, tc.get("id"))
            except Exception as e:
                return (idx, None, e, tool_name, tc.get("id"))

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
        for idx, out_json, err, tool_name, tool_call_id in sorted(results, key=lambda x: x[0]):
            if err is not None:
                err_s = str(err)
                conv.append({"role": "tool", "tool_call_id": tool_call_id, "content": f"error: {err_s}"})
                inv_recs.append({"name": tool_name, "ok": False, "error": err_s})
            else:
                try:
                    inv = json.loads(out_json) if isinstance(out_json, str) else (out_json or {})
                except Exception:
                    inv = {"ok": False, "error": "invalid tool JSON"}
                text = inv.get("result") or inv.get("answer") or inv.get("text") or inv.get("stdout") or ""
                conv.append({"role": "tool", "tool_call_id": tool_call_id, "content": text})
                inv_recs.append({"name": tool_name, **inv})

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
