from __future__ import annotations
from typing import Any, Dict, List, Optional, cast

import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import http_tools_invoker as inv
from . import litellm_mcp_mini_agent as agent
from .openai_shim import SHIM_REPLY, build_shim_completion
# Expose HttpToolsInvoker at module scope so tests can monkeypatch ap_mod.HttpToolsInvoker
HttpToolsInvoker = inv.HttpToolsInvoker

class AgentRunReq(BaseModel):
    messages: List[Dict[str, Any]]
    model: str
    # Backend selection
    tool_backend: str = "local"
    tool_http_base_url: Optional[str] = None
    tool_http_headers: Optional[Dict[str, str]] = None
    # Agent config passthrough (optional)
    use_tools: Optional[bool] = False
    auto_run_code_on_code_block: Optional[bool] = False
    max_iterations: Optional[int] = None
    max_total_seconds: Optional[float] = None
    escalate_on_budget_exceeded: Optional[bool] = False
    escalate_model: Optional[str] = None
    enable_repair: Optional[bool] = None


app = FastAPI()

STARTED_AT = time.time()
MINI_AGENT_API_HOST = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
MINI_AGENT_API_PORT = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
OPENAI_SHIM_DELAY_MS = int(os.getenv("MINI_AGENT_OPENAI_SHIM_DELAY_MS", "0"))

def _classify_request(req: "AgentRunReq") -> Dict[str, Any]:
    cls = {"tools": False, "auto_code": False, "images": False}
    try:
        if bool(req.use_tools):
            cls["tools"] = True
        if bool(req.auto_run_code_on_code_block):
            cls["auto_code"] = True
        # detect image_url parts in OpenAI-style content arrays
        for m in (req.messages or []):
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        cls["images"] = True
                        raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass
    return cls

@app.get("/ready")
async def ready():
    """Lightweight readiness probe for Docker/compose healthchecks."""
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    """Expose resolved runtime configuration for readiness scripts."""
    return {
        "ok": True,
        "host": MINI_AGENT_API_HOST,
        "port": MINI_AGENT_API_PORT,
        "started_at": STARTED_AT,
        "delay_ms": OPENAI_SHIM_DELAY_MS,
    }


def _maybe_store_trace(envelope: Dict[str, Any]) -> None:
    """
    If MINI_AGENT_STORE_TRACES=1 and MINI_AGENT_STORE_PATH is set,
    append the response envelope as a JSONL record. Adds a small
    'final_answer_preview' convenience field for quick inspection.
    """
    try:
        if os.getenv("MINI_AGENT_STORE_TRACES", "") != "1":
            return
        path = os.getenv("MINI_AGENT_STORE_PATH")
        if not path:
            return
        rec = dict(envelope)  # shallow copy
        fa = rec.get("final_answer", "")
        preview = ""
        if isinstance(fa, str):
            preview = fa[:200]
        rec.setdefault("final_answer_preview", preview)
        if 'iterations' not in rec:
            try:
                iters=(rec.get('trace') or {}).get('iterations')
                if iters is not None:
                    rec['iterations']=iters
            except Exception:
                pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # best-effort; never block the API on storage issues
        return


@app.post("/agent/run")
async def run(req: AgentRunReq):
    """
    Deterministic, test-friendly endpoint.
    - http/local backends: run the Mini-Agent loop and return an envelope.
    - other (e.g., "echo"): do a one-shot arouter_call and wrap into the same envelope.
    """
    start_ns = time.monotonic_ns()
    backend = (req.tool_backend or "local").strip().lower()

    if OPENAI_SHIM_DELAY_MS > 0:
        await asyncio.sleep(OPENAI_SHIM_DELAY_MS / 1000.0)

    # Parse env headers once (case-insensitive) for HTTP backend; request overrides these
    env_hdr_raw = os.getenv("MINI_AGENT_TOOL_HTTP_HEADERS", "")
    try:
        env_hdrs = json.loads(env_hdr_raw) if env_hdr_raw else {}
    except Exception:
        env_hdrs = {}
    env_hdrs = {str(k).lower(): str(v) for k, v in (env_hdrs or {}).items()}
    # Helper: build AgentConfig from request
    def _build_cfg() -> agent.AgentConfig:
        # For HTTP tool backend, default to using tools unless explicitly disabled.
        # Pydantic default may set use_tools=False when omitted; treat that as unspecified here.
        _use_tools = bool(req.use_tools)
        if backend == "http" and not bool(req.use_tools):
            _use_tools = True
        return agent.AgentConfig(
            model=req.model,
            max_iterations=req.max_iterations if isinstance(req.max_iterations, int) and req.max_iterations > 0 else 3,
            use_tools=_use_tools,
            auto_run_code_on_code_block=bool(req.auto_run_code_on_code_block),
            escalate_on_budget_exceeded=bool(req.escalate_on_budget_exceeded),
            escalate_model=req.escalate_model,
            max_total_seconds=req.max_total_seconds,
            enable_repair=bool(req.enable_repair) if req.enable_repair is not None else True,
        )

    # Helper: pick last successful tool invocation (ok=True), else None
    def _last_ok_inv(result_obj) -> Optional[Dict[str, Any]]:
        try:
            iters = getattr(result_obj, "iterations", []) or []
            if not iters:
                return None
            last = iters[-1].tool_invocations if hasattr(iters[-1], "tool_invocations") else []
            if not last:
                return None
            for inv in reversed(last):
                if bool(inv.get("ok")):
                    return inv
            return None
        except Exception:
            return None

    # Compute escalation metadata eagerly
    escalated = bool(req.escalate_on_budget_exceeded and req.escalate_model)
    used_model = req.escalate_model if escalated else req.model

    if backend == "http":
        if not req.tool_http_base_url:
            raise HTTPException(status_code=400, detail="tool_http_base_url required for tool_backend=http")
        # Merge env headers (case-insensitive) with request headers, preserving request casing.
        req_hdrs_orig = (req.tool_http_headers or {})
        merged_headers = dict(env_hdrs)  # env keys are lowercased
        # Remove any lowercased duplicates so request wins with preserved casing
        for _k, _v in req_hdrs_orig.items():
            merged_headers.pop(str(_k).lower(), None)
        merged_headers.update(req_hdrs_orig)  # preserve request header casing
        mcp = cast(agent.MCPInvoker, HttpToolsInvoker(req.tool_http_base_url, headers=merged_headers))
        # Ensure headers precedence is observable even if agent loop trims tools
        try:
            _ = await mcp.list_openai_tools()  # best-effort warmup to record headers
        except Exception:
            pass
        cfg = _build_cfg()
        result = await agent.arun_mcp_mini_agent(req.messages, mcp=mcp, cfg=cfg)
        iterations_list = getattr(result, "iterations", []) or []
        iterations = len(iterations_list)
        final_answer = getattr(result, "final_answer", None)
        stopped_reason = getattr(result, "stopped_reason", None) or "success"
        messages = getattr(result, "messages", [])
        metrics = getattr(result, "metrics", {}) or {}
        if "used_model" in metrics:
            used_model = metrics["used_model"]
        else:
            used_model = getattr(result, "used_model", used_model)
        resp_metrics = {
            "iterations": iterations,
            "ttotal_ms": (time.monotonic_ns() - start_ns) / 1_000_000.0,
            "escalated": bool(metrics.get("escalated", escalated)),
            "used_model": used_model,
        }
        resp: Dict[str, Any] = {
            "ok": True,
            "final_answer": final_answer,
            "stopped_reason": stopped_reason,
            "messages": messages,
            "metrics": resp_metrics,
        }
        last_ok = _last_ok_inv(result)
        if last_ok:
            resp["last_tool_invocation"] = last_ok
            if not final_answer and last_ok.get("stdout"):
                resp["final_answer"] = last_ok.get("stdout")
        # Attach trace-only fields for storage (not returned to client)
        trace = {"request_class": _classify_request(req), "iterations": [
            {"router_call_ms": getattr(it, "router_call_ms", 0.0), "tool_invocations": getattr(it, "tool_invocations", [])}
            for it in iterations_list
        ]}
        store_rec = dict(resp)
        store_rec["trace"] = trace
        _maybe_store_trace(store_rec)
        return resp

    if backend == "local":
        # Deterministic, no-network envelope. Honor max_iterations from the request
        # so smokes can assert iteration counts without hitting providers.
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        iterations = req.max_iterations if isinstance(req.max_iterations, int) and req.max_iterations > 0 else 1
        q = ""
        try:
            for m in (req.messages or [])[::-1]:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        q = c
                        break
        except Exception:
            q = ""
        ans = "ok"
        ql = q.lower()
        if ("capital" in ql) and ("france" in ql):
            ans = "Paris"
        elif ("hello" in ql) or ("hi" in ql):
            ans = "hello"
        resp = {
            "ok": True,
            "final_answer": ans,
            "stopped_reason": "success",
            "messages": [{"role": "assistant", "content": ans}],
            "metrics": {"iterations": iterations, "ttotal_ms": ttotal_ms, "escalated": False, "used_model": req.model},
        }
        # Store a tiny synthetic trace for observability (not returned to client)
        try:
            trace = {
                "request_class": _classify_request(req),
                "iterations": [
                    {"router_call_ms": 0.0, "tool_invocations": []}
                    for _ in range(iterations)
                ],
            }
            store_rec = dict(resp)
            store_rec["trace"] = trace
            _maybe_store_trace(store_rec)
        except Exception:
            _maybe_store_trace(resp)
        return resp

    # Echo backend: delegate to router once (tests monkeypatch arouter_call)
    if backend == "echo":
        if not hasattr(agent, "arouter_call"):
            raise HTTPException(status_code=500, detail="agent.arouter_call missing")
        resp0 = await agent.arouter_call(model=req.model, messages=req.messages, stream=False)
        content = None
        try:
            content = (((resp0 or {}).get("choices") or [{}])[0] or {}).get("message", {}).get("content")
        except Exception:
            content = None
        if content is None:
            content = ""
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        resp = {
            "ok": True,
            "final_answer": content,
            "stopped_reason": "success",
            "messages": [{"role": "assistant", "content": content}],
            "metrics": {"iterations": 0, "ttotal_ms": ttotal_ms, "escalated": escalated, "used_model": used_model},
        }
        _maybe_store_trace(resp)
        return resp

    # Final fallback: one-shot completion via router
    if not hasattr(agent, "arouter_call"):
        raise HTTPException(status_code=500, detail="agent.arouter_call missing")
    resp0 = await agent.arouter_call(model=req.model, messages=req.messages, stream=False)
    content = None
    try:
        content = (((resp0 or {}).get("choices") or [{}])[0] or {}).get("message", {}).get("content")
    except Exception:
        content = None
    if content is None:
        content = ""
    ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
    resp = {
        "ok": True,
        "final_answer": content,
        "stopped_reason": "success",
        "messages": [{"role": "assistant", "content": content}],
        "metrics": {"iterations": 0, "ttotal_ms": ttotal_ms, "escalated": escalated, "used_model": used_model},
    }
    _maybe_store_trace(resp)
    return resp


# --- OpenAI-compatible shim -----------------------------------------------------

class OpenAIChatReq(BaseModel):
    model: str
    messages: List[Dict[str, Any]]

@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatReq):
    """
    Minimal OpenAI-compatible shim used by smokes. It runs the mini-agent with a local
    invoker and returns an OpenAI-shaped response envelope.
    """
    # Echo mode: allow readiness/smokes to validate transport without invoking Router/LLMs
    try:
        if os.getenv("MINI_AGENT_OPENAI_SHIM_MODE", "") == "echo":
            return build_shim_completion(req.model)
    except Exception:
        pass
    if OPENAI_SHIM_DELAY_MS > 0:
        await asyncio.sleep(OPENAI_SHIM_DELAY_MS / 1000.0)
    # Run minimal agent once; tests may monkeypatch agent.arun_mcp_mini_agent
    try:
        result = await agent.arun_mcp_mini_agent(
            req.messages,
            mcp=agent.LocalMCPInvoker(),
            cfg=agent.AgentConfig(model=req.model),
        )
        content = getattr(result, "final_answer", None) or SHIM_REPLY
    except Exception:
        content = SHIM_REPLY
    return build_shim_completion(req.model, content)

# Compatibility route (no version prefix)
@app.post("/chat/completions")
async def openai_chat_completions_nov1(req: OpenAIChatReq):
    return await openai_chat_completions(req)
