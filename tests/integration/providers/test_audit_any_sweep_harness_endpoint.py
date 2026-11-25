from __future__ import annotations

import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from pydantic import JsonValue

from litellm.harness.context import GatewayTarget
from litellm.harness.endpoint import ModelEndpoint
from litellm.harness.types import Harness


def _register_scenario(gateway: Gateway, scenario_id: str, response: dict[str, JsonValue]) -> None:
    with httpx.Client(timeout=5, trust_env=False) as client:
        result: Final = client.post(
            f"{gateway.upstream_url}/__scenarios",
            json={"scenario_id": scenario_id, "response": response},
        )
    assert result.status_code == 200, result.text


def _client(endpoint: ModelEndpoint) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=endpoint.url,
        headers={"Authorization": f"Bearer {endpoint.token}"},
        trust_env=False,
        timeout=15,
    )


_CHAT_BODY: Final = {
    "id": "chatcmpl-$UNIQUE_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "scripted-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}

_RESPONSES_BODY: Final = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "status": "completed",
    "model": "scripted-model",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


async def test_endpoint_forwards_json_object_bodies(gateway: Gateway) -> None:
    scenario_id: Final = f"audithend{uuid.uuid4().hex[:8]}"
    _register_scenario(
        gateway,
        scenario_id,
        {
            "content_type": "application/x-routed",
            "routes": {
                "POST /v1/chat/completions": {"content_type": "application/json", "body": _CHAT_BODY},
                "POST /v1/responses": {"content_type": "application/json", "body": _RESPONSES_BODY},
            },
        },
    )
    target: Final = GatewayTarget(api_base=f"{gateway.upstream_url}/{scenario_id}", api_key="scripted")
    async with ModelEndpoint(Harness.CODEX, "audit-model", target, api_key="x") as endpoint:
        assert isinstance(endpoint.port, int) and endpoint.port > 0
        async with _client(endpoint) as client:
            chat: Final = await client.post("/v1/chat/completions", json={"messages": []})
            assert chat.status_code == 200, chat.text
            assert chat.json()["object"] == "chat.completion", chat.text
            responses: Final = await client.post("/v1/responses", json={"input": "x"})
            assert responses.status_code == 200, responses.text
            assert responses.json()["object"] == "response", responses.text
    usage: Final = endpoint.usage.snapshot()
    assert usage.input_tokens == 11 + 5 and usage.output_tokens == 7 + 3, usage


@pytest.mark.parametrize("raw", (b"[1, 2]", b'"a string"', b"42", b"null"))
async def test_endpoint_rejects_valid_json_non_object(gateway: Gateway, raw: bytes) -> None:
    async with ModelEndpoint(Harness.CODEX, "audit-model", None, api_key="x") as endpoint:
        async with _client(endpoint) as client:
            response: Final = await client.post("/v1/chat/completions", content=raw)
            assert response.status_code == 400, response.text
            assert "must be a JSON object" in response.text, response.text


async def test_endpoint_rejects_unparseable_body(gateway: Gateway) -> None:
    async with ModelEndpoint(Harness.CODEX, "audit-model", None, api_key="x") as endpoint:
        async with _client(endpoint) as client:
            response: Final = await client.post("/v1/chat/completions", content=b"this is not json{")
            assert response.status_code == 400, response.text


async def test_endpoint_records_usage_from_sse_stream(gateway: Gateway) -> None:
    scenario_id: Final = f"audithend{uuid.uuid4().hex[:8]}"
    _register_scenario(
        gateway,
        scenario_id,
        {
            "content_type": "text/event-stream",
            "frames": (
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}',
                "data: 42",
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":13,"completion_tokens":4}}',
                "data: [DONE]",
            ),
        },
    )
    target: Final = GatewayTarget(api_base=f"{gateway.upstream_url}/{scenario_id}", api_key="scripted")
    async with ModelEndpoint(Harness.CODEX, "audit-model", target, api_key="x") as endpoint:
        async with _client(endpoint) as client:
            response: Final = await client.post("/v1/chat/completions", json={"messages": []})
            assert response.status_code == 200, response.text
            assert "[DONE]" in response.text, response.text
    usage: Final = endpoint.usage.snapshot()
    assert usage.input_tokens == 13 and usage.output_tokens == 4, usage


async def test_endpoint_binds_free_port() -> None:
    async with ModelEndpoint(Harness.CODEX, "audit-model", None) as endpoint:
        port: Final = endpoint.port
        assert isinstance(port, int) and port > 0
        async with _client(endpoint) as client:
            response: Final = await client.get("/v1/models")
            assert response.status_code == 200, response.text
            assert response.json()["object"] == "list", response.text
