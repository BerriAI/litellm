# LiteLLM Readiness Smoke Review — 2025-09-28

## Summary
- Strict readiness run: `. .venv/bin/activate && READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=all_smokes_core python scripts/mvp_check.py`
- Failure: `tests/smoke_optional/test_agent_proxy_validation_errors.py::test_agent_proxy_headers_precedence` because `HttpToolsInvoker` stays patched after a prior smoke.
- Root cause: `tests/smoke_optional/test_agent_proxy_headers_and_errors.py` overwrites `HttpToolsInvoker` without restoring it, so the next test never hits the real header merge path.

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
@@ test_agent_proxy_headers_and_errors.py (leaking monkeypatch) @@
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
+ 
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

## Fix Recommendation
1. Use `monkeypatch.setattr(ap_mod, "HttpToolsInvoker", _FakeInvoker)` (auto-restore) or save/restore manually.
2. Re-run the strict readiness command above and confirm green.
3. Wire `smokes-nd-real` with Ollama into deploy readiness to restore ND variance.

Full code context follows in this gist.