import json
import types
import sys
import pytest


class _Resp:
    def __init__(self, json_obj, status_code=200):
        self._json = json_obj
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):  # noqa: A002
        if url.endswith("/responses"):
            # Emulate Perplexity responses endpoint
            return _Resp({
                "output": [{"content": "Answer with evidence"}],
                "citations": [{"title": "Doc", "url": "https://example.com"}],
            })
        return _Resp({}, 404)

    async def get(self, url, params=None, headers=None):
        if url.endswith("/search"):
            return _Resp({"snippets": [{"path": "x.md", "excerpt": "foo"}]})
        return _Resp({}, 404)


@pytest.mark.smoke
def test_research_python_tools_basic(monkeypatch):
    # Provide fake httpx
    # Provide a minimal httpx surface for our invoker and avoid importing litellm's httpx
    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    # Also stub problematic imports inside litellm's custom_httpx layer
    monkeypatch.setitem(sys.modules, "httpx._config", types.SimpleNamespace(DEFAULT_LIMITS=None))
    monkeypatch.setitem(sys.modules, "httpx._transports.default", types.SimpleNamespace(AsyncHTTPTransport=object, HTTPTransport=object))
    monkeypatch.setitem(sys.modules, "httpx._client", types.SimpleNamespace(USE_CLIENT_DEFAULT=None))

    # Import tool invoker (avoid importing top-level litellm by loading module via path)
    import importlib.util
    # Provide a minimal stub for MCPInvoker to satisfy relative import without importing heavy litellm
    import types as _types
    stub_ma = _types.ModuleType("litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent")
    class _MCPInvoker:
        async def list_openai_tools(self):
            raise NotImplementedError
        async def call_openai_tool(self, openai_tool):
            raise NotImplementedError
    stub_ma.MCPInvoker = _MCPInvoker
    # also stub arouter_call used by research_perplexity
    async def _fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"content": "Answer with evidence"}}]}
    stub_ma.arouter_call = _fake_arouter_call
    sys.modules["litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent"] = stub_ma

    spec = importlib.util.spec_from_file_location(
        "litellm.experimental_mcp_client.mini_agent.research_tools",
        "litellm/experimental_mcp_client/mini_agent/research_tools.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    ResearchPythonInvoker = mod.ResearchPythonInvoker

    # Provide envs for invoker
    monkeypatch.setenv("PPLX_API_KEY", "fake")
    monkeypatch.setenv("C7_API_BASE", "https://c7.example.com")
    inv = ResearchPythonInvoker()
    import asyncio

    # Perplexity
    t_perp = {
        "id": "tc",
        "type": "function",
        "function": {"name": "research_perplexity", "arguments": json.dumps({"query": "q"})},
    }
    out1 = asyncio.run(inv.call_openai_tool(t_perp))
    data1 = json.loads(out1)
    assert data1.get("ok") is True and data1.get("citations")

    # Context7
    t_ctx = {
        "id": "tc2",
        "type": "function",
        "function": {"name": "research_context7_docs", "arguments": json.dumps({"library": "lib", "topic": "t"})},
    }
    out2 = asyncio.run(inv.call_openai_tool(t_ctx))
    data2 = json.loads(out2)
    assert data2.get("ok") is True and data2.get("snippets")
