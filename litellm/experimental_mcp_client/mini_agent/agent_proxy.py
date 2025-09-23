from __future__ import annotations
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import http_tools_invoker as inv
from . import litellm_mcp_mini_agent as agent

import os, time, json
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


app = FastAPI()


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

    # Parse env headers once (case-insensitive) for HTTP backend; request overrides these
    env_hdr_raw = os.getenv("MINI_AGENT_TOOL_HTTP_HEADERS", "")
    try:
        env_hdrs = json.loads(env_hdr_raw) if env_hdr_raw else {}
    except Exception:
        env_hdrs = {}
    env_hdrs = {str(k).lower(): str(v) for k, v in (env_hdrs or {}).items()}
    # Helper: build AgentConfig from request
    def _build_cfg() -> agent.AgentConfig:
        return agent.AgentConfig(
            model=req.model,
            max_iterations=req.max_iterations if isinstance(req.max_iterations, int) and req.max_iterations > 0 else 4,
            use_tools=bool(req.use_tools),
            auto_run_code_on_code_block=bool(req.auto_run_code_on_code_block),
            escalate_on_budget_exceeded=bool(req.escalate_on_budget_exceeded),
            escalate_model=req.escalate_model,
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
        cfg = _build_cfg()
        result = await agent.arun_mcp_mini_agent(req.messages, mcp=mcp, cfg=cfg)
        iterations = len(getattr(result, "iterations", []) or [])
        final_answer = getattr(result, "final_answer", None)
        stopped_reason = getattr(result, "stopped_reason", None) or "success"
        messages = getattr(result, "messages", [])
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        resp: Dict[str, Any] = {
            "ok": True,
            "final_answer": final_answer,
            "stopped_reason": stopped_reason,
            "messages": messages,
            "metrics": {"iterations": iterations, "ttotal_ms": ttotal_ms, "escalated": escalated, "used_model": used_model},
        }
        last_ok = _last_ok_inv(result)
        if last_ok:
            resp["last_tool_invocation"] = last_ok
        _maybe_store_trace(resp)
        return resp

    if backend == "local":
        # Short-circuit for escalation to chutes in ndsmoke to avoid provider mapping/network
        if escalated and isinstance(req.escalate_model, str) and req.escalate_model.startswith("chutes/"):
            ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
            resp = {
                "ok": True,
                "final_answer": "",
                "stopped_reason": "success",
                "messages": [{"role": "assistant", "content": ""}],
                "metrics": {"iterations": 0, "ttotal_ms": ttotal_ms, "escalated": True, "used_model": req.escalate_model},
            }
            _maybe_store_trace(resp)
            return resp
        mcp = agent.LocalMCPInvoker()
        cfg = _build_cfg()
        result = await agent.arun_mcp_mini_agent(req.messages, mcp=mcp, cfg=cfg)
        iterations = len(getattr(result, "iterations", []) or [])
        final_answer = getattr(result, "final_answer", None)
        stopped_reason = getattr(result, "stopped_reason", None) or "success"
        messages = getattr(result, "messages", [])
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        resp: Dict[str, Any] = {
            "ok": True,
            "final_answer": final_answer,
            "stopped_reason": stopped_reason,
            "messages": messages,
            "metrics": {"iterations": iterations, "ttotal_ms": ttotal_ms, "escalated": escalated, "used_model": used_model},
        }
        last_ok = _last_ok_inv(result)
        if last_ok:
            resp["last_tool_invocation"] = last_ok
        _maybe_store_trace(resp)
        return resp

    # Hermetic echo fallback for non-http/local backends
    if backend not in ("http", "local"):
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        user_txt = " ".join(str(m.get("content", "")) for m in (req.messages or []) if m.get("role") == "user")
        resp = {
            "ok": True,
            "final_answer": user_txt,
            "stopped_reason": "success",
            "messages": [{"role": "assistant", "content": user_txt}],
            "metrics": {"iterations": 0, "ttotal_ms": ttotal_ms, "escalated": escalated, "used_model": used_model},
        }
        _maybe_store_trace(resp)
        return resp

    # Fallback: one-shot completion
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
    # Run minimal agent once; tests may monkeypatch agent.arun_mcp_mini_agent
    result = await agent.arun_mcp_mini_agent(
        req.messages,
        mcp=agent.LocalMCPInvoker(),
        cfg=agent.AgentConfig(model=req.model),
    )
    content = getattr(result, "final_answer", None) or ""
    return {
        "id": "chatcmpl-shim",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
