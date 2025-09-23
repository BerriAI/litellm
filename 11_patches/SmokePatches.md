Yes—ship it as a single unified diff. That’s the cleanest way for review/CI, and it lets you `git apply` in one shot.

Below is a ready-to-apply patch that:

1. switches relevant smokes to `pytest-asyncio` (no manual loops),
2. removes the custom session loop fixture, and
3. adds a handful of high-value new smokes.

> Assumes repo root at `litellm/`. If your test folder path differs, tweak the paths.

```diff
*** Begin Patch
*** Add File: pytest.ini
+ [pytest]
+ asyncio_mode = auto
+
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/conftest.py
@@
-import asyncio
 import os
 import shutil
 import sys
-from typing import Generator
 
 import pytest
-
-
-@pytest.fixture(scope="session")
-def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
-    """
-    Provide a single asyncio event loop for smoke tests.
-    Ensures async tests (mini-agent, parallel fan-out) run deterministically.
-    """
-    loop = asyncio.new_event_loop()
-    try:
-        yield loop
-    finally:
-        loop.close()
@@
 def have_codex_env() -> bool:
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_agent_escalate_smoke.py
@@
-import asyncio
 import pytest
 
-@pytest.mark.smoke
-def test_agent_escalate_on_last_step(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_agent_escalate_on_last_step(monkeypatch):
     from litellm.experimental_mcp_client.mini_agent import litellm_mcp_mini_agent as agent
@@
-    res = asyncio.get_event_loop().run_until_complete(
-        agent.arun_mcp_mini_agent(messages, mcp=agent.EchoMCP(), cfg=cfg)
-    )
+    res = await agent.arun_mcp_mini_agent(messages, mcp=agent.EchoMCP(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_batch_as_completed_ordering.py
@@
-import asyncio
 import pytest
@@
-@pytest.mark.smoke
-def test_batch_as_completed_ordering():
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_batch_as_completed_ordering():
     from litellm.extras.batch import acompletion_as_completed
@@
-    outs = asyncio.run(run())
+    outs = await run()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_chutes_stub_smoke.py
@@
-import asyncio
 import pytest
@@
-@pytest.mark.smoke
-def test_chutes_chat_stubbed(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_chutes_chat_stubbed(monkeypatch):
@@
-    out = asyncio.get_event_loop().run_until_complete(run())
+    out = await run()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_http_error_tail.py
@@
-import types
-import pytest
+import types
+import pytest
@@
-@pytest.mark.smoke
-def test_http_tools_invoker_error_tail(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_http_tools_invoker_error_tail(monkeypatch):
@@
-    import asyncio
-    with pytest.raises(Exception) as ei:
-        asyncio.run(inv.call_openai_tool(call))
+    with pytest.raises(Exception) as ei:
+        await inv.call_openai_tool(call)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_http_tools_invoker.py
@@
-@pytest.mark.smoke
-def test_http_tools_invoker_monkeypatch(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_http_tools_invoker_monkeypatch(monkeypatch):
@@
-    tools = asyncio.run(inv.list_openai_tools())
+    tools = await inv.list_openai_tools()
@@
-    out = asyncio.run(inv.call_openai_tool(openai_call))
+    out = await inv.call_openai_tool(openai_call)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_http_tools_invoker_headers.py
@@
-@pytest.mark.smoke
-def test_http_tools_invoker_headers_passthrough(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_http_tools_invoker_headers_passthrough(monkeypatch):
@@
-    asyncio.run(inv.list_openai_tools())
+    await inv.list_openai_tools()
@@
-    asyncio.run(inv.call_openai_tool({"function": {"name": "echo", "arguments": "{}"}}))
+    await inv.call_openai_tool({"function": {"name": "echo", "arguments": "{}"}})
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_images_local_remote.py
@@
-import asyncio
@@
-@pytest.mark.smoke
-def test_fetch_remote_image_cache(monkeypatch, tmp_path):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_fetch_remote_image_cache(monkeypatch, tmp_path):
@@
-    data_url1 = asyncio.run(run())
-    data_url2 = asyncio.run(run())
+    data_url1 = await run()
+    data_url2 = await run()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_local_docs_litellm_call_image_prep.py
@@
-import asyncio
 import pytest
@@
-@pytest.mark.smoke
-def test_litellm_call_image_prep_uses_utils(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_litellm_call_image_prep_uses_utils(monkeypatch):
@@
-    out = asyncio.run(prep(msgs, image_cache_dir=None))
+    out = await prep(msgs, image_cache_dir=None)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_codeblock_autorun_smoke.py
@@
-import asyncio
 import json
 import pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_autorun_python_codeblock_smoke(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_autorun_python_codeblock_smoke(monkeypatch):
@@
-    out = asyncio.run(agent.arun_mcp_mini_agent(
-        messages=[{"role": "user", "content": "do it"}], mcp=StubMCP(), cfg=cfg
-    ))
+    out = await agent.arun_mcp_mini_agent(
+        messages=[{"role": "user", "content": "do it"}], mcp=StubMCP(), cfg=cfg
+    )
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_codeblock_multilang_smoke.py
@@
-import asyncio, json, pytest
+import json, pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_autorun_c_codeblock_smoke(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_autorun_c_codeblock_smoke(monkeypatch):
@@
-    out = asyncio.run(agent.arun_mcp_mini_agent(messages=[{"role":"user","content":"do it"}], mcp=StubMCP(), cfg=cfg))
+    out = await agent.arun_mcp_mini_agent(messages=[{"role":"user","content":"do it"}], mcp=StubMCP(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_codex_optional.py
@@
-import os, asyncio, pytest
+import os, pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_codex_optional(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_codex_optional(monkeypatch):
@@
-    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=EchoMCP(), cfg=cfg))
+    out = await arun_mcp_mini_agent(messages, mcp=EchoMCP(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_loop_isolation_smoke.py
@@
-import json
-import types
-import sys
-import pytest
+import json, types, sys, pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_loop_isolation_until_clean(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_loop_isolation_until_clean(monkeypatch):
@@
-    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg))
+    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_repair_loop.py
@@
-import json
-import types
-import sys
-import pytest
+import json, types, sys, pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_repair_loop_exec_python(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_repair_loop_exec_python(monkeypatch):
@@
-    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg))
+    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_mini_agent_research_on_unsure.py
@@
-import json
-import types
-import sys
-import pytest
+import json, types, sys, pytest
@@
-@pytest.mark.smoke
-def test_mini_agent_research_on_unsure(monkeypatch):
+@pytest.mark.smoke
+@pytest.mark.asyncio
+async def test_mini_agent_research_on_unsure(monkeypatch):
@@
-    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg))
+    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_router_builder_smoke.py
@@
-import asyncio
 import pytest
@@
-def test_router_builder_param_propagation(monkeypatch):
+@pytest.mark.asyncio
+async def test_router_builder_param_propagation(monkeypatch):
@@
-    resp = asyncio.get_event_loop().run_until_complete(router.acompletion(
-        model="alias-a",
-        messages=[{"role": "user", "content": "hi"}],
-        temperature=0.33,
-        tool_choice="auto",
-        timeout=1.23,
-    ))
+    resp = await router.acompletion(
+        model="alias-a",
+        messages=[{"role": "user", "content": "hi"}],
+        temperature=0.33,
+        tool_choice="auto",
+        timeout=1.23,
+    )
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_router_streaming_smoke.py
@@
-import asyncio
-import os
-import types
-import pytest
+import asyncio, os, types, pytest
@@
-@pytest.mark.parametrize("mode", ["legacy", "extracted"])
-def test_router_streaming_smoke(monkeypatch, mode):
+@pytest.mark.parametrize("mode", ["legacy", "extracted"])
+@pytest.mark.asyncio
+async def test_router_streaming_smoke(monkeypatch, mode):
@@
-    stream = asyncio.get_event_loop().run_until_complete(router.acompletion(
-        model="alias-a",
-        messages=[{"role": "user", "content": "hi"}],
-        stream=True,
-    ))
+    stream = await router.acompletion(
+        model="alias-a",
+        messages=[{"role": "user", "content": "hi"}],
+        stream=True,
+    )
@@
-    async def _do():
-        async for chunk in stream:
-            if chunk and getattr(chunk.choices[0].delta, "content", None):
-                collected.append(chunk.choices[0].delta.content)
-    asyncio.get_event_loop().run_until_complete(_do())
+    async for chunk in stream:
+        if chunk and getattr(chunk.choices[0].delta, "content", None):
+            collected.append(chunk.choices[0].delta.content)
*** End Patch
```

```diff
*** Begin Patch
*** Update File: litellm/tests/smoke/test_router_teardown_smoke.py
@@
-import asyncio
 import pytest
@@
-def test_router_teardown_smoke(monkeypatch):
+@pytest.mark.asyncio
+async def test_router_teardown_smoke(monkeypatch):
@@
-    resp = asyncio.get_event_loop().run_until_complete(router.acompletion(
-        model="alias-a", messages=[{"role": "user", "content": "hi"}]
-    ))
+    resp = await router.acompletion(
+        model="alias-a", messages=[{"role": "user", "content": "hi"}]
+    )
*** End Patch
```

---

## New smoke tests

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_router_fallback_non_streaming_parity.py
+ import pytest
+ from litellm.router import Router
+ 
+ class _Resp:
+     def __init__(self, text: str): self.choices=[type("C",(),{"message":type("M",(),{"content":text})()})()]
+ 
+ @pytest.mark.smoke
+ @pytest.mark.asyncio
+ async def test_router_fallback_non_streaming_parity(monkeypatch):
+     r = Router(model_list=[{"model_name":"m","litellm_params":{"model":"openai/gpt-4o-mini","api_key":"sk"}}])
+ 
+     calls = {"primary":0,"fb":0}
+     async def primary(**kw):
+         calls["primary"]+=1
+         raise RuntimeError("primary boom")
+     async def fb(*a,**k):
+         calls["fb"]+=1
+         return _Resp("ok-fallback")
+ 
+     import litellm
+     monkeypatch.setattr(litellm,"acompletion",primary)
+     monkeypatch.setattr(Router,"async_function_with_fallbacks_common_utils",staticmethod(lambda *a,**k: fb()))
+ 
+     out = await r.acompletion(model="m", messages=[{"role":"user","content":"hi"}], stream=False)
+     try:
+         txt = out.choices[0].message.content
+     except Exception:
+         txt = out.get("choices",[{}])[0].get("message",{}).get("content")
+     assert txt == "ok-fallback"
+     assert calls["primary"]>=1 and calls["fb"]==1
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_parallel_acompletions_exceptions_and_limits.py
+ import asyncio
+ import pytest
+ from litellm import Router
+ from litellm.router_utils.parallel_acompletion import RouterParallelRequest
+ 
+ @pytest.mark.asyncio
+ async def test_parallel_acompletions_exceptions_and_limits(monkeypatch):
+     r = Router()
+     async def stub(model, messages, **kw):
+         txt = messages[0]["content"]
+         await asyncio.sleep(0)
+         if "ERR" in txt:
+             raise RuntimeError("boom")
+         return type("R",(),{"text":f"ok:{txt}"})
+     monkeypatch.setattr(r,"acompletion",stub)
+ 
+     reqs = [
+         RouterParallelRequest("m",[{"role":"user","content":"A"}],{}),
+         RouterParallelRequest("m",[{"role":"user","content":"ERR"}],{}),
+         RouterParallelRequest("m",[{"role":"user","content":"B"}],{}),
+     ]
+     out = await r.parallel_acompletions(reqs, preserve_order=True, return_exceptions=True, concurrency=10)
+     assert [o.index for o in out]==[0,1,2]
+     assert out[0].error is None and out[2].error is None
+     assert isinstance(out[1].error, Exception)
+ 
+ @pytest.mark.asyncio
+ async def test_parallel_acompletions_empty_list():
+     r = Router()
+     out = await r.parallel_acompletions([], preserve_order=True)
+     assert out == []
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_agent_proxy_validation_errors.py
+ import pytest
+ fastapi = pytest.importorskip("fastapi")
+ from fastapi.testclient import TestClient
+ 
+ @pytest.mark.smoke
+ def test_agent_proxy_validation_errors(monkeypatch):
+     from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
+     c = TestClient(app)
+     # Missing required fields
+     r = c.post("/agent/run", json={"model":"x"})
+     assert r.status_code in (400,422)
+     data = r.json()
+     assert isinstance(data, dict)
+ 
+ @pytest.mark.smoke
+ def test_agent_proxy_headers_precedence(monkeypatch):
+     from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
+     from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
+     from fastapi.testclient import TestClient
+     recorded={}
+     class _Client:
+         def __init__(self, *a, headers=None, **k): recorded["headers"]=headers or {}
+         async def __aenter__(self): return self
+         async def __aexit__(self,*a): return False
+         async def get(self, url): return type("R",(),{"json":lambda self: [] ,"raise_for_status":lambda self:None})()
+         async def post(self, url, json=None): return type("R",(),{"json":lambda self:{"text":"ok"},"raise_for_status":lambda self:None})()
+     import types
+     inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
+     client = TestClient(app)
+     monkeypatch.setenv("MINI_AGENT_TOOL_HTTP_HEADERS",'{"X-Env":"A"}')
+     r = client.post("/agent/run", json={
+         "messages":[{"role":"user","content":"hi"}],
+         "model":"m","tool_backend":"http","tool_http_base_url":"http://127.0.0.1:9",
+         "tool_http_headers":{"X-Env":"B","X-Req":"C"}
+     })
+     assert r.status_code==200
+     h = recorded.get("headers",{})
+     assert h.get("X-Env")=="B" and h.get("X-Req")=="C"
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_mini_agent_tool_args_non_json.py
+ import json, pytest, sys, types
+ 
+ @pytest.mark.smoke
+ @pytest.mark.asyncio
+ async def test_mini_agent_tool_args_non_json(monkeypatch):
+     monkeypatch.setitem(sys.modules,"fastuuid", types.SimpleNamespace(uuid4=lambda:"0"*32))
+     monkeypatch.setitem(sys.modules,"mcp", types.SimpleNamespace(ClientSession=object))
+     monkeypatch.setitem(sys.modules,"mcp.types", types.SimpleNamespace(
+         CallToolRequestParams=object, CallToolResult=object, Tool=object))
+     from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent
+     import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod
+ 
+     async def fake_call(**kw):
+         return {"choices":[{"message":{"role":"assistant","content":"tool","tool_calls":[
+             {"id":"tc","type":"function","function":{"name":"echo","arguments":"{BAD JSON]"}}
+         ]}}]}
+     monkeypatch.setattr(mod,"arouter_call",fake_call)
+ 
+     out = await arun_mcp_mini_agent([{"role":"user","content":"go"}], mcp=LocalMCPInvoker(), cfg=AgentConfig(model="m",max_iterations=2,enable_repair=True,use_tools=True))
+     # Should not crash; an observation about parse error should be appended and loop continues
+     joined = "\n".join((m.get("content") or "") for m in out.messages if isinstance(m,dict))
+     assert "Observation" in joined or out.stopped_reason in ("success","max_iterations")
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_http_tools_invoker_non_json_body.py
+ import types, pytest
+ 
+ @pytest.mark.smoke
+ @pytest.mark.asyncio
+ async def test_http_tools_invoker_non_json_body(monkeypatch):
+     from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
+     class _Client:
+         async def __aenter__(self): return self
+         async def __aexit__(self,*a): return False
+         async def get(self, url): return type("R",(),{"json":lambda s: [],"raise_for_status":lambda s:None})()
+         async def post(self, url, json=None):
+             return type("R",(),{
+                 "raise_for_status":lambda s: None,
+                 "json": lambda s: (_ for _ in ()).throw(ValueError("not json")),
+                 "text": "plain text body",
+                 "status_code": 200
+             })()
+     inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
+     inv = inv_mod.HttpToolsInvoker("http://fake")
+     out = await inv.call_openai_tool({"function":{"name":"echo","arguments":"{\"text\":\"x\"}"}})
+     assert isinstance(out, str)
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_prune_history_budget_smaller_than_pair.py
+ import pytest, sys, types
+ 
+ @pytest.mark.smoke
+ def test_prune_history_budget_smaller_than_pair(monkeypatch):
+     sys.modules.setdefault("fastuuid", types.SimpleNamespace(uuid4=lambda:"0"*32))
+     sys.modules.setdefault("mcp", types.SimpleNamespace(ClientSession=object))
+     sys.modules.setdefault("mcp.types", types.SimpleNamespace(CallToolRequestParams=object,CallToolResult=object,Tool=object))
+     import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod
+     messages=[
+         {"role":"system","content":"s"},
+         {"role":"user","content":"u"*100},
+         {"role":"assistant","content":None,"tool_calls":[{"id":"tc","type":"function","function":{"name":"echo","arguments":"{}"}}]},
+         {"role":"tool","tool_call_id":"tc","content":"X"*200},
+     ]
+     pruned = mod._prune_history_preserve_pair(messages, max_non_system=1, hard_char_budget=10)
+     assert any(m.get("role")=="assistant" and mod._get_tool_calls(m) for m in pruned)
+     assert any(m.get("role")=="tool" and m.get("tool_call_id")=="tc" for m in pruned)
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_research_missing_envs_graceful_skip.py
+ import json, pytest, importlib.util
+ 
+ @pytest.mark.smoke
+ def test_research_missing_envs_graceful_skip(monkeypatch):
+     spec = importlib.util.spec_from_file_location(
+         "litellm.experimental_mcp_client.mini_agent.research_tools",
+         "litellm/experimental_mcp_client/mini_agent/research_tools.py",
+     )
+     mod = importlib.util.module_from_spec(spec)
+     assert spec and spec.loader
+     spec.loader.exec_module(mod)  # type: ignore[attr-defined]
+     inv = mod.ResearchPythonInvoker()
+     # Unset envs
+     monkeypatch.delenv("PPLX_API_KEY", raising=False)
+     monkeypatch.delenv("C7_API_BASE", raising=False)
+     import asyncio
+     out = asyncio.run(inv.call_openai_tool({"function":{"name":"research_perplexity","arguments":json.dumps({"query":"x"})}}))
+     data = json.loads(out)
+     assert data.get("ok") is False or "error" in data
*** End Patch
```

```diff
*** Begin Patch
*** Add File: litellm/tests/smoke/test_batch_as_completed_return_exceptions.py
+ import pytest, asyncio
+ from litellm.extras.batch import acompletion_as_completed
+ 
+ class _R:
+     async def acompletion(self, *, model, messages, **kw):
+         txt = messages[0]["content"]
+         await asyncio.sleep(0)
+         if "ERR" in txt:
+             raise RuntimeError("boom")
+         return {"choices":[{"message":{"content":"OK"}}]}
+ 
+ @pytest.mark.asyncio
+ async def test_as_completed_return_exceptions():
+     reqs = [
+         {"model":"m","messages":[{"role":"user","content":"ok"}]},
+         {"model":"m","messages":[{"role":"user","content":"ERR"}]},
+     ]
+     outs=[]
+     async for idx, resp in acompletion_as_completed(_R(), reqs, concurrency=2, return_exceptions=True):
+         outs.append((idx, resp))
+     assert len(outs)==2 and any(isinstance(o[1], Exception) for o in outs)
*** End Patch
```

---

### Notes

* I didn’t touch files that were already async or that never used a manual loop.
* If you’ve got a `pyproject.toml` test extra, include `pytest-asyncio>=0.23`. Otherwise your CI workflow can `pip install pytest-asyncio`.
* If any provider code relies on a **session-scoped** event loop, we can re-introduce one via `pytest_asyncio.fixture(scope="session")`, but your current smokes don’t need it.

If you want me to include *every* converted test file in one super-patch (including all the ndsmokes you didn’t need to change), say the word and I’ll dump the full diff.
