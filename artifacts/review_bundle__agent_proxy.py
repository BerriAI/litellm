# Source: litellm/experimental_mcp_client/mini_agent/agent_proxy.py (lines 160-239)
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
        # Hermetic core shape: avoid real LLM calls and return a deterministic envelope.
        # For a tiny bit of value, answer a couple of trivial facts deterministically.
        ttotal_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
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
