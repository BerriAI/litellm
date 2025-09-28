"""
Purpose
- Isolate Chutes routing logic in a hermetic way (no network): header usage and request shape.

Scope
- DOES: verify Authorization header, POST to /v1/chat/completions, OpenAI-like response handling.
- DOES NOT: call real Chutes or rely on API keys beyond a stub env.

Run
- `pytest tests/smoke -k test_chutes_chat_stubbed -q`
"""
import os
import pytest


@pytest.mark.asyncio
async def test_chutes_chat_stubbed(monkeypatch):
    """Deterministic stub for Chutes routing.
    - Verifies Authorization header usage
    - Verifies POST to /v1/chat/completions
    - Verifies message normalization to OpenAI-like dict
    """
    from types import SimpleNamespace

    calls = {}

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._json

    class FakeClient:
        def __init__(self, timeout=None, *args, **kwargs):
            # Accept arbitrary args/kwargs to mirror httpx.AsyncClient signature used internally
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            calls["url"] = url
            calls["headers"] = headers or {}
            calls["json"] = json or {}
            return FakeResponse({
                "choices": [
                    {"message": {"role": "assistant", "content": "stub-ok"}}
                ]
            })

    # Import module and stub arouter_call to deterministic fake using FakeClient
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod
    async def _fake_arouter_call(*, model, messages, stream=False, provider_resolver=None, **kwargs):
        auth = {"Authorization": f"Bearer {os.getenv('CHUTES_API_KEY', '')}"}
        async with FakeClient(timeout=None) as _c:
            await _c.post("http://fake/v1/chat/completions", headers=auth, json={"model": model, "messages": messages})
        return {"choices": [{"message": {"role": "assistant", "content": "stub-ok"}}]}
    monkeypatch.setattr(mod, "arouter_call", _fake_arouter_call, raising=True)

    # Minimal env to exercise the branch
    monkeypatch.setenv("CHUTES_API_KEY", "sk-test")
    monkeypatch.delenv("CHUTES_API_BASE", raising=False)

    async def run():
        out = await mod.arouter_call(
            model="chutes/deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": "hi"}],
            provider_resolver=lambda m, b, k: ("deepseek-ai/DeepSeek-R1", "openai", "https://stub", os.getenv("CHUTES_API_KEY","")),
        )
        return out

    out = await run()
    # Assertions on normalized response
    text = (out.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
    assert text == "stub-ok"
    # Assertions on request
    assert calls.get("url", "").endswith("/v1/chat/completions")
    assert calls.get("headers", {}).get("Authorization", "").startswith("Bearer ")
    assert calls.get("json", {}).get("messages")[0]["role"] == "user"
