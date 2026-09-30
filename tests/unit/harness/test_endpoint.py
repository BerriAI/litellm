import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import litellm
from litellm.harness import endpoint as endpoint_module
from litellm.harness.endpoint import (
    ModelEndpoint,
    SSEUsageParser,
    UsageTracker,
    compute_cost,
    usage_from_body,
)
from litellm.harness.errors import HarnessInstallFailed
from litellm.harness.context import GatewayTarget
from litellm.harness.types import Harness, Usage
from litellm.types.utils import ModelResponse, ModelResponseStream

GATEWAY = GatewayTarget(api_base="https://gw.example.com", api_key="sk-gateway-secret")

ANTHROPIC_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)
CHAT_SSE = (
    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n'
    b"data: [DONE]\n\n"
)
RESPONSES_SSE = (
    b"event: response.output_text.delta\n"
    b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    b"event: response.completed\n"
    b'data: {"type":"response.completed","response":{"usage":{"input_tokens":20,"output_tokens":4}}}\n\n'
)


class Recorder:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


def sse_response(body: bytes, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": "text/event-stream", **(headers or {})},
    )


def gateway_endpoint(recorder: Recorder, **kwargs: Any) -> ModelEndpoint:
    return ModelEndpoint(
        Harness.CLAUDE_CODE,
        kwargs.pop("model", "claude-sonnet"),
        GATEWAY,
        transport=httpx.MockTransport(recorder),
        **kwargs,
    )


def auth(ep: ModelEndpoint) -> dict[str, str]:
    return {"authorization": f"Bearer {ep.token}"}


@pytest.fixture(autouse=True)
def no_real_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda model, prompt_tokens, completion_tokens: (
            prompt_tokens * 0.001,
            completion_tokens * 0.002,
        ),
    )


async def test_rejects_bad_token_and_accepts_both_header_styles() -> None:
    recorder = Recorder(httpx.Response(200, json={"usage": {}}))
    async with gateway_endpoint(recorder) as ep:
        assert ep.url == f"http://127.0.0.1:{ep.port}" and ep.port > 0
        async with httpx.AsyncClient(base_url=ep.url) as client:
            missing = await client.post("/v1/messages", json={})
            wrong = await client.post(
                "/v1/messages", json={}, headers={"x-api-key": "nope"}
            )
            bearer = await client.post("/v1/messages", json={}, headers=auth(ep))
            api_key = await client.post(
                "/messages", json={}, headers={"x-api-key": ep.token}
            )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert "error" in wrong.json()
    assert bearer.status_code == 200
    assert api_key.status_code == 200
    assert len(recorder.requests) == 2


async def test_gateway_rewrites_headers_and_model() -> None:
    recorder = Recorder(
        httpx.Response(
            200,
            json={"id": "m", "usage": {"input_tokens": 3, "output_tokens": 2}},
        )
    )
    async with gateway_endpoint(recorder, metadata={"run": "abc"}) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post(
                "/v1/messages",
                json={"model": "whatever", "max_tokens": 5},
                headers={
                    "x-api-key": ep.token,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "tools-2024",
                },
            )
    assert resp.status_code == 200
    sent = recorder.requests[0]
    assert str(sent.url) == "https://gw.example.com/v1/messages"
    assert sent.headers["authorization"] == "Bearer sk-gateway-secret"
    assert "x-api-key" not in sent.headers
    assert sent.headers["x-litellm-tags"] == "harness,claude_code"
    assert json.loads(sent.headers["x-litellm-spend-logs-metadata"]) == {"run": "abc"}
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.headers["anthropic-beta"] == "tools-2024"
    assert json.loads(sent.content)["model"] == "claude-sonnet"
    assert ep.usage.input_tokens == 3 and ep.usage.output_tokens == 2
    assert ep.usage.calls == 1


@pytest.mark.parametrize(
    "path,body,expected",
    [
        ("/v1/messages", ANTHROPIC_SSE, (11, 7)),
        ("/v1/chat/completions", CHAT_SSE, (5, 3)),
        ("/responses", RESPONSES_SSE, (20, 4)),
    ],
)
async def test_gateway_sse_passthrough_and_usage(
    path: str, body: bytes, expected: tuple[int, int]
) -> None:
    recorder = Recorder(sse_response(body))
    async with gateway_endpoint(recorder) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post(path, json={"stream": True}, headers=auth(ep))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == body
    assert (ep.usage.input_tokens, ep.usage.output_tokens) == expected
    expected_cost = expected[0] * 0.001 + expected[1] * 0.002
    assert ep.usage.cost == pytest.approx(expected_cost)


async def test_cost_header_preferred_over_computed() -> None:
    recorder = Recorder(
        httpx.Response(
            200,
            json={"usage": {"prompt_tokens": 100, "completion_tokens": 100}},
            headers={"x-litellm-response-cost": "0.42"},
        )
    )
    async with gateway_endpoint(recorder) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            await client.post("/v1/chat/completions", json={}, headers=auth(ep))
    assert ep.usage.cost == pytest.approx(0.42)
    assert ep.usage.snapshot() == Usage(input_tokens=100, output_tokens=100, calls=1)


async def test_gateway_error_status_preserved_and_not_counted() -> None:
    recorder = Recorder(httpx.Response(429, json={"error": "rate limited"}))
    async with gateway_endpoint(recorder) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post("/v1/chat/completions", json={}, headers=auth(ep))
    assert resp.status_code == 429
    assert ep.usage.calls == 0


async def test_models_route() -> None:
    recorder = Recorder(httpx.Response(200))
    async with gateway_endpoint(recorder) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            with_model = await client.get("/v1/models", headers=auth(ep))
            unauth = await client.get("/models")
    assert unauth.status_code == 401
    assert with_model.json()["object"] == "list"
    assert [m["id"] for m in with_model.json()["data"]] == ["claude-sonnet"]

    async with ModelEndpoint(Harness.CODEX, None, None) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            empty = await client.get("/models", headers=auth(ep))
    assert empty.json() == {"object": "list", "data": []}


async def test_sdk_chat_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        calls.append(kwargs)
        response = ModelResponse(
            model="gpt-x",
            choices=[{"message": {"role": "assistant", "content": "hello"}}],
            usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        )
        response._hidden_params["response_cost"] = 0.5
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    async with ModelEndpoint(
        Harness.OPENCODE, "openai/gpt-x", None, api_key="sk-real", api_base="https://x"
    ) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "ignored",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=auth(ep),
            )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello"
    assert calls[0]["model"] == "openai/gpt-x"
    assert calls[0]["api_key"] == "sk-real"
    assert calls[0]["api_base"] == "https://x"
    assert (ep.usage.input_tokens, ep.usage.output_tokens) == (9, 4)
    assert ep.usage.cost == pytest.approx(0.5)


async def fake_chat_stream() -> AsyncIterator[ModelResponseStream]:
    yield ModelResponseStream(choices=[{"delta": {"content": "he"}}])
    yield ModelResponseStream(choices=[{"delta": {"content": "llo"}}])
    final = ModelResponseStream(choices=[])
    final.usage = litellm.Usage(prompt_tokens=6, completion_tokens=2, total_tokens=8)
    yield final


async def test_sdk_chat_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[ModelResponseStream]:
        calls.append(kwargs)
        return fake_chat_stream()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    async with ModelEndpoint(Harness.OPENCODE, "openai/gpt-x", None) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post(
                "/chat/completions",
                json={"messages": [], "stream": True},
                headers=auth(ep),
            )
    assert resp.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[0][6:])["choices"][0]["delta"]["content"] == "he"
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert (ep.usage.input_tokens, ep.usage.output_tokens) == (6, 2)
    assert ep.usage.cost == pytest.approx(6 * 0.001 + 2 * 0.002)


async def fake_anthropic_stream() -> AsyncIterator[Any]:
    yield {"type": "message_start", "message": {"usage": {"input_tokens": 4}}}
    yield b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'


async def test_sdk_messages_stream_handles_dicts_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acreate(**kwargs: Any) -> AsyncIterator[Any]:
        return fake_anthropic_stream()

    monkeypatch.setattr(litellm.anthropic.messages, "acreate", fake_acreate)
    async with ModelEndpoint(Harness.CLAUDE_CODE, "anthropic/claude", None) as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post(
                "/v1/messages", json={"stream": True}, headers=auth(ep)
            )
    assert "event: message_start" in resp.text
    assert "event: message_delta" in resp.text
    assert (ep.usage.input_tokens, ep.usage.output_tokens) == (4, 9)


async def test_sdk_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing(**kwargs: Any) -> Any:
        raise litellm.RateLimitError(
            message="too many requests for key sk-real",
            llm_provider="openai",
            model="gpt-x",
        )

    monkeypatch.setattr(litellm, "aresponses", failing)
    async with ModelEndpoint(Harness.CODEX, "gpt-x", None, api_key="sk-real") as ep:
        async with httpx.AsyncClient(base_url=ep.url) as client:
            resp = await client.post("/v1/responses", json={}, headers=auth(ep))
    assert resp.status_code == 429
    assert "sk-real" not in resp.text
    assert resp.json()["error"]["type"] == "RateLimitError"


async def test_missing_server_deps_raises_install_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise HarnessInstallFailed(endpoint_module.MISSING_DEPS_MESSAGE)

    monkeypatch.setattr(endpoint_module, "_load_server_deps", missing)
    with pytest.raises(HarnessInstallFailed, match="pip install starlette uvicorn"):
        async with ModelEndpoint(Harness.CODEX, None, None):
            pass


def test_load_server_deps_maps_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(HarnessInstallFailed, match="starlette and uvicorn"):
        endpoint_module._load_server_deps()


def test_usage_helpers() -> None:
    assert usage_from_body({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}) == (
        1,
        2,
    )
    assert usage_from_body({"response": {"usage": {"input_tokens": 3}}}) == (3, 0)
    assert usage_from_body("nope") == (0, 0)

    parser = SSEUsageParser()
    for i in range(0, len(ANTHROPIC_SSE), 7):  # split across arbitrary chunk borders
        parser.feed(ANTHROPIC_SSE[i : i + 7])
    parser.close()
    assert (parser.input_tokens, parser.output_tokens) == (11, 7)

    tracker = UsageTracker()
    tracker.add(1, 2, 0.1)
    tracker.add(3, 4, 0.2)
    assert tracker.snapshot() == Usage(input_tokens=4, output_tokens=6, calls=2)
    assert tracker.cost == pytest.approx(0.3)


def test_compute_cost_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: Any) -> Any:
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm, "cost_per_token", boom)
    assert compute_cost("mystery", 10, 10) == 0.0
    assert compute_cost(None, 10, 10) == 0.0
