# LiteLLM Readiness Smoke Review — 2025-09-28
## Overall Assessment
- Strict readiness command: `. .venv/bin/activate && READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=all_smokes_core python scripts/mvp_check.py` (red).
- Core deterministic tests pass; failure confined to `all_smokes_core` smoke suite.
- ND-real coverage remains absent unless `ND_REAL=1`, so deploy readiness currently lacks variance checks.

## Failures & Evidence
- `tests/smoke_optional/test_agent_proxy_validation_errors.py::test_agent_proxy_headers_precedence` expects merged headers `{'x-env': 'B', 'x-req': 'C'}` but records `{}`.
- `tests/smoke_optional/test_agent_proxy_headers_and_errors.py` runs earlier in the suite and replaces `HttpToolsInvoker`; it appears the override persists into the failing test.
- Running the failing test in isolation inside `.venv` passes, supporting the shared-state hypothesis.

## Work Attempted So Far
- Reproduced the readiness failure with the command above.
- Executed the failing test alone (`pytest tests/smoke_optional/test_agent_proxy_validation_errors.py::...`) — passes.
- Inspected `agent_proxy.py` and `http_tools_invoker.py`; header merge logic seems correct when the genuine invoker is used.
- Identified the potential source of leakage in `tests/smoke_optional/test_agent_proxy_headers_and_errors.py`; no changes applied yet.

## Help Requested
- Looking for reviewer guidance: does the analysis hold, and what specific changes should be made (if any) to scope the prior smoke's monkeypatch and restore `ND` coverage expectations?
- Please share blunt feedback and preferred unified diffs so we can course-correct correctly.

## Key Diff Context
```diff
@@ agent_proxy.py (HTTP backend header merge) @@
+     if backend == "http":
+         if not req.tool_http_base_url:
+             raise HTTPException(status_code=400, detail="tool_http_base_url required for tool_backend=http")
+         # Merge env headers (case-insensitive) with request headers, preserving request casing.
+         req_hdrs_orig = (req.tool_http_headers or {})
+         merged_headers = dict(env_hdrs)  # env keys are lowercased
+         # Remove any lowercased duplicates so request wins with preserved casing
+         for _k, _v in req_hdrs_orig.items():
+             merged_headers.pop(str(_k).lower(), None)
+         merged_headers.update(req_hdrs_orig)  # preserve request header casing
+         mcp = cast(agent.MCPInvoker, HttpToolsInvoker(req.tool_http_base_url, headers=merged_headers))
+         # Ensure headers precedence is observable even if agent loop trims tools
+         try:
+             _ = await mcp.list_openai_tools()  # best-effort warmup to record headers
+         except Exception:
@@ http_tools_invoker.py (AsyncClient usage) @@
+ class HttpToolsInvoker:
+     def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None) -> None:
+         self.base_url = base_url.rstrip("/")
+         self.headers = headers or {}
+ 
+     def _mk_client(self):
+         """
+         Create an AsyncClient. Some test doubles don't accept kwargs; fall back gracefully.
+         """
+         AsyncClient = getattr(httpx, "AsyncClient", object)
+         try:
+             return AsyncClient(headers=self.headers)  # type: ignore[call-arg]
+         except TypeError:
+             return AsyncClient()  # type: ignore[call-arg]
+ 
+     async def list_openai_tools(self) -> List[Dict[str, Any]]:
+         async with self._mk_client() as client:  # type: ignore
+             # Prefer passing headers; fall back for stubs that don't accept it
+             try:
+                 r = await client.get(f"{self.base_url}/tools", headers=self.headers)
+             except TypeError:
+                 r = await client.get(f"{self.base_url}/tools")
+             if getattr(r, "status_code", 200) >= 400:
+                 tail = (getattr(r, "text", "") or "")[:256]
+                 raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")
+             try:
+                 return r.json()
+             except Exception:
+                 return []
+ 
+     async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
+         fn = (openai_tool or {}).get("function", {})
+         name = fn.get("name", "")
+         args = fn.get("arguments", {})
+         body = {"name": name, "arguments": args}
+ 
+         async with self._mk_client() as client:  # type: ignore
+             
+             try:
+                 r = await client.post(f"{self.base_url}/invoke", json=body, headers=self.headers)
+             except TypeError:
+                 # test doubles may not accept keyword args; fall back
+                 r = await client.post(f"{self.base_url}/invoke", body)
+ 
+ 
+             # One polite retry on 429 honoring Retry-After if present
+             if getattr(r, "status_code", 200) == 429:
+                 ra = None
+                 try:
+                     ra = getattr(r, "headers", {}).get("Retry-After") if hasattr(r, "headers") else None
+                 except Exception:
+                     ra = None
+                 try:
+                     delay = float(ra) if ra is not None else 0.0
@@ test_agent_proxy_headers_and_errors.py (potential leak) @@
+     # Replace HttpToolsInvoker with a dummy that records headers and tools
+     recorded = {}
+ 
+     class _FakeInvoker:
+         def __init__(self, base_url, headers=None):
+             recorded["base_url"] = base_url
+             recorded["headers"] = headers or {}
+         async def list_openai_tools(self):
+             return []
+         async def call_openai_tool(self, openai_tool):
+             return json.dumps({"ok": True, "text": "hi"})
+ 
+     ap_mod.HttpToolsInvoker = _FakeInvoker
+ 
+     client = TestClient(app)
+ 
+     # 1) Missing tool_http_base_url → 400
+     r = client.post("/agent/run", json={
+         "messages": [{"role": "user", "content": "test"}],
+         "model": "dummy",
+         "tool_backend": "http"
+     })
+     assert r.status_code == 400
+ 
+     # 2) Headers passthrough
+     r2 = client.post("/agent/run", json={
+         "messages": [{"role": "user", "content": "test"}],
+         "model": "dummy",
+         "tool_backend": "http",
+         "tool_http_base_url": "http://127.0.0.1:9999",
+         "tool_http_headers": {"Authorization": "Bearer X"}
+     })
+     assert r2.status_code == 200
+     assert recorded.get("headers", {}).get("Authorization") == "Bearer X"
@@ test_agent_proxy_validation_errors.py (contract) @@
+     inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
+ 
+     # Short-circuit the LLM call; we only care that /tools (and/or /invoke) was called with the right headers
+     async def _fake_router_call(*, model, messages, stream=False, **kwargs):
+         return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
+     monkeypatch.setattr(mini, "arouter_call", _fake_router_call, raising=True)
+ 
+     # Set env headers; request headers should override these
+     monkeypatch.setenv("MINI_AGENT_TOOL_HTTP_HEADERS", json.dumps({"X-Env": "A"}))
+ 
+     c = TestClient(app)
+     r = c.post(
+         "/agent/run",
+         json={
+             "messages": [{"role": "user", "content": "hi"}],
+             "model": "m",
+             "tool_backend": "http",
+             "tool_http_base_url": "http://127.0.0.1:9",
+             "tool_http_headers": {"X-Env": "B", "X-Req": "C"},
+         },
+     )
+     assert r.status_code == 200
+     # Normalize header keys to lowercase for case-insensitive compare
+     hdrs = {str(k).lower(): v for k, v in (recorded.get("headers", {}) or {}).items()}
+     assert hdrs.get("x-env") == "B" and hdrs.get("x-req") == "C"
```

## Code Context
### litellm/experimental_mcp_client/mini_agent/agent_proxy.py (lines 170-224)
```python
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
```
### litellm/experimental_mcp_client/mini_agent/http_tools_invoker.py (lines 1-140)
```python
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional

import types as _types
try:
    import httpx  # type: ignore
except Exception:
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("httpx is required for HttpToolsInvoker HTTP calls")

        async def post(self, *args, **kwargs):
            raise RuntimeError("httpx is required for HttpToolsInvoker HTTP calls")

    httpx = _types.SimpleNamespace(AsyncClient=_DummyAsyncClient)  # type: ignore


class HttpToolsInvoker:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    def _mk_client(self):
        """
        Create an AsyncClient. Some test doubles don't accept kwargs; fall back gracefully.
        """
        AsyncClient = getattr(httpx, "AsyncClient", object)
        try:
            return AsyncClient(headers=self.headers)  # type: ignore[call-arg]
        except TypeError:
            return AsyncClient()  # type: ignore[call-arg]

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        async with self._mk_client() as client:  # type: ignore
            # Prefer passing headers; fall back for stubs that don't accept it
            try:
                r = await client.get(f"{self.base_url}/tools", headers=self.headers)
            except TypeError:
                r = await client.get(f"{self.base_url}/tools")
            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")
            try:
                return r.json()
            except Exception:
                return []

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        fn = (openai_tool or {}).get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        body = {"name": name, "arguments": args}

        async with self._mk_client() as client:  # type: ignore
            
            try:
                r = await client.post(f"{self.base_url}/invoke", json=body, headers=self.headers)
            except TypeError:
                # test doubles may not accept keyword args; fall back
                r = await client.post(f"{self.base_url}/invoke", body)


            # One polite retry on 429 honoring Retry-After if present
            if getattr(r, "status_code", 200) == 429:
                ra = None
                try:
                    ra = getattr(r, "headers", {}).get("Retry-After") if hasattr(r, "headers") else None
                except Exception:
                    ra = None
                try:
                    delay = float(ra) if ra is not None else 0.0
                except Exception:
                    delay = 0.0
                # cap small delay to avoid slow tests
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))
                
            try:
                r = await client.post(f"{self.base_url}/invoke", json=body, headers=self.headers)
            except TypeError:
                # test doubles may not accept keyword args; fall back
                r = await client.post(f"{self.base_url}/invoke", body)


            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")

            # Prefer JSON; if not JSON, return plain text body
            try:
                data = r.json()
            except Exception:
                txt = getattr(r, "text", "")
                return txt if isinstance(txt, str) else str(txt)

            if isinstance(data, dict):
                # common keys returned by simple tool executors
                for k in ("text", "result", "answer"):
                    v = data.get(k)
                    if isinstance(v, str):
                        return v
                return json.dumps(data, ensure_ascii=False)

            return json.dumps(data, ensure_ascii=False)
```
### tests/smoke_optional/test_agent_proxy_headers_and_errors.py (lines 1-120)
```python
import json
import sys
import types
import pytest


@pytest.mark.smoke
def test_agent_proxy_errors_and_headers(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    _ = pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    # Stub optional deps pulled by litellm on import
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    monkeypatch.setitem(sys.modules, "mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(
        sys.modules,
        "mcp.types",
        types.SimpleNamespace(
            CallToolRequestParams=type("CallToolRequestParams", (), {}),
            CallToolResult=type("CallToolResult", (), {}),
            Tool=type("Tool", (), {}),
        ),
    )

    # Import app now
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.agent_proxy as ap_mod
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent_mod

    # Monkeypatch arouter_call to short-circuit LLM call
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    agent_mod.arouter_call = fake_arouter_call

    # Missing httpx -> 400
    class _FakeInvokerMissing:  # raises ImportError scenario is handled at call-site already
        pass

    # Replace HttpToolsInvoker with a dummy that records headers and tools
    recorded = {}

    class _FakeInvoker:
        def __init__(self, base_url, headers=None):
            recorded["base_url"] = base_url
            recorded["headers"] = headers or {}
        async def list_openai_tools(self):
            return []
        async def call_openai_tool(self, openai_tool):
            return json.dumps({"ok": True, "text": "hi"})

    ap_mod.HttpToolsInvoker = _FakeInvoker

    client = TestClient(app)

    # 1) Missing tool_http_base_url → 400
    r = client.post("/agent/run", json={
        "messages": [{"role": "user", "content": "test"}],
        "model": "dummy",
        "tool_backend": "http"
    })
    assert r.status_code == 400

    # 2) Headers passthrough
    r2 = client.post("/agent/run", json={
        "messages": [{"role": "user", "content": "test"}],
        "model": "dummy",
        "tool_backend": "http",
        "tool_http_base_url": "http://127.0.0.1:9999",
        "tool_http_headers": {"Authorization": "Bearer X"}
    })
    assert r2.status_code == 200
    assert recorded.get("headers", {}).get("Authorization") == "Bearer X"

```
### tests/smoke_optional/test_agent_proxy_validation_errors.py (lines 1-120)
```python
# tests/smoke/test_agent_proxy_validation_errors.py
import json
import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

@pytest.mark.smoke
def test_agent_proxy_headers_precedence(monkeypatch):
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    from litellm.experimental_mcp_client.mini_agent import litellm_mcp_mini_agent as mini
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app

    recorded = {}

    class _Client:
        def __init__(self, *a, headers=None, **k):
            if headers is not None:
                recorded["headers"] = dict(headers)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None):
            if headers is not None:
                recorded["headers"] = dict(headers)
            return type("R", (), {"json": lambda self: [], "raise_for_status": lambda self: None})()
        async def post(self, url, json=None, headers=None):
            if headers is not None:
                recorded["headers"] = dict(headers)
            return type("R", (), {"json": lambda self: {"text": "ok"}, "raise_for_status": lambda self: None})()

    # Use our tolerant httpx stub so we can capture headers
    import types
    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)

    # Short-circuit the LLM call; we only care that /tools (and/or /invoke) was called with the right headers
    async def _fake_router_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr(mini, "arouter_call", _fake_router_call, raising=True)

    # Set env headers; request headers should override these
    monkeypatch.setenv("MINI_AGENT_TOOL_HTTP_HEADERS", json.dumps({"X-Env": "A"}))

    c = TestClient(app)
    r = c.post(
        "/agent/run",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "m",
            "tool_backend": "http",
            "tool_http_base_url": "http://127.0.0.1:9",
            "tool_http_headers": {"X-Env": "B", "X-Req": "C"},
        },
    )
    assert r.status_code == 200
    # Normalize header keys to lowercase for case-insensitive compare
    hdrs = {str(k).lower(): v for k, v in (recorded.get("headers", {}) or {}).items()}
    assert hdrs.get("x-env") == "B" and hdrs.get("x-req") == "C"

```
