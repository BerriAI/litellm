"""
Purpose
- Research tools degrade gracefully when required env vars are missing.

Scope
- DOES: unset PPLX/C7 envs; assert call returns {ok:false} or an 'error' field.
- DOES NOT: make real network calls (uses stubbed module load).

Run
- `pytest tests/smoke -k test_research_missing_envs_graceful_skip -q`
"""
import json, pytest, importlib.util, asyncio

def test_research_missing_envs_graceful_skip(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "litellm.experimental_mcp_client.mini_agent.research_tools",
        "litellm/experimental_mcp_client/mini_agent/research_tools.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    inv = mod.ResearchPythonInvoker()
    # Unset envs
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("C7_API_BASE", raising=False)
    out = asyncio.run(inv.call_openai_tool({"function":{"name":"research_perplexity","arguments":json.dumps({"query":"x"})}}))
    data = json.loads(out)
    assert data.get("ok") is False or "error" in data
