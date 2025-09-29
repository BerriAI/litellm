import httpx
import pytest


@pytest.mark.asyncio
async def test_codex_agent_sidecar_autostart(monkeypatch):
    from litellm.llms import codex_agent

    calls = {}

    def fake_ensure():
        calls.setdefault("ensure", 0)
        calls["ensure"] += 1
        return "http://127.0.0.1:9999"

    monkeypatch.setattr(codex_agent, "ensure_sidecar", fake_ensure)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(self, url, json, headers=None, timeout=None):
        return FakeResponse()

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    llm = codex_agent.CodexAgentLLM()
    from litellm.utils import ModelResponse

    resp = llm.completion(
        model="codex-agent/gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda *a, **k: None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={},
        headers={},
        timeout=None,
    )

    assert resp.choices[0].message.content == "hello"
    assert calls.get("ensure") == 1
