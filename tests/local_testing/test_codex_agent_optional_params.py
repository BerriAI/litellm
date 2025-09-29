import pytest

from litellm.llms.codex_agent import CodexAgentLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import ModelResponse


class _DummySyncResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _DummyHTTPHandler(HTTPHandler):
    def __init__(self):
        self.requests = []

    def post(self, url, data=None, json=None, params=None, headers=None, stream=False, timeout=None, files=None, content=None, logging_obj=None):  # type: ignore[override]
        self.requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
                "stream": stream,
            }
        )
        return _DummySyncResponse({"choices": [{"message": {"content": "ok"}}]})


class _DummyAsyncResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _DummyAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self):
        self.requests = []

    async def post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, logging_obj=None, files=None, content=None):  # type: ignore[override]
        self.requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
                "stream": stream,
            }
        )
        return _DummyAsyncResponse({"choices": [{"message": {"content": "async"}}]})


def _mk_llm(monkeypatch):
    monkeypatch.setenv("CODEX_AGENT_API_BASE", "http://codex.local")
    return CodexAgentLLM()


def test_codex_agent_completion_forwards_optional_params(monkeypatch):
    llm = _mk_llm(monkeypatch)
    handler = _DummyHTTPHandler()

    resp = llm.completion(
        model="codex-agent/demo",
        messages=[{"role": "user", "content": "hi"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda *a, **k: None,
        encoding=None,
        api_key="token-123",
        logging_obj=None,
        optional_params={"temperature": 0.2, "stream": False},
        headers={"X-Test": "1"},
        timeout=5.0,
        client=handler,
    )

    assert resp.choices[0].message.content == "ok"
    assert handler.requests, "expected HTTP handler to be invoked"
    sent = handler.requests[0]
    assert sent["json"]["temperature"] == 0.2
    assert sent["headers"]["Authorization"] == "Bearer token-123"
    assert sent["headers"]["X-Test"] == "1"


@pytest.mark.asyncio
async def test_codex_agent_acompletion_uses_async_handler(monkeypatch):
    llm = _mk_llm(monkeypatch)
    handler = _DummyAsyncHTTPHandler()

    resp = await llm.acompletion(
        model="codex-agent/demo",
        messages=[{"role": "user", "content": "async"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda *a, **k: None,
        encoding=None,
        api_key="token-xyz",
        logging_obj=None,
        optional_params={"max_tokens": 33},
        headers={},
        timeout=10.0,
        client=handler,
    )

    assert resp.choices[0].message.content == "async"
    assert handler.requests, "expected async HTTP handler to be invoked"
    sent = handler.requests[0]
    assert sent["json"]["max_tokens"] == 33
    assert sent["headers"]["Authorization"] == "Bearer token-xyz"
