"""
Near-E2E tests for A2A 0.3/1.0 version routing through the proxy.

Runs invoke_agent_a2a -> asend_message -> a2a-sdk 1.x -> ASGI mock upstream.
Only proxy auth/registry/pre-call plumbing is patched; version normalization
runs on the real response path.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

pytest.importorskip("a2a.compat.v0_3.types")

from litellm.proxy._types import UserAPIKeyAuth

UPSTREAM_BASE = "http://testserver"

_UPSTREAM_CALLS: List[Dict[str, Any]] = []


def _upstream_card_payload() -> Dict[str, Any]:
    return {
        "protocolVersion": "0.3",
        "name": "mock-agent",
        "url": f"{UPSTREAM_BASE}/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _message_result(request_id: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "kind": "message",
            "role": "agent",
            "messageId": "m-out",
            "parts": [{"kind": "text", "text": "pong"}],
        },
    }


def _sse_stream(request_id: Any) -> AsyncIterator[bytes]:
    events = [
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "task",
                "id": "t1",
                "contextId": "c1",
                "status": {"state": "submitted"},
            },
        },
        _message_result(request_id),
    ]

    async def _gen() -> AsyncIterator[bytes]:
        for event in events:
            yield f"data: {json.dumps(event)}\n\n".encode()

    return _gen()


async def _serve_upstream_agent_card(request: Any) -> JSONResponse:
    return JSONResponse(_upstream_card_payload())


async def _upstream_jsonrpc(request: Any) -> JSONResponse | StreamingResponse:
    body = await request.json()
    _UPSTREAM_CALLS.append(body)
    request_id = body.get("id", "req-1")
    method = body.get("method")

    if method == "message/stream":
        return StreamingResponse(
            _sse_stream(request_id),
            media_type="text/event-stream",
        )

    return JSONResponse(_message_result(request_id))


def _build_upstream_app() -> Starlette:
    return Starlette(
        routes=[
            Route(
                "/.well-known/agent-card.json",
                _serve_upstream_agent_card,
                methods=["GET"],
            ),
            Route("/.well-known/agent.json", _serve_upstream_agent_card, methods=["GET"]),
            Route("/", _upstream_jsonrpc, methods=["POST"]),
        ]
    )


def _fake_get_async_httpx_client(
    llm_provider: Any = None, params: Optional[Dict[str, Any]] = None
) -> MagicMock:
    handler = MagicMock()
    handler.client = httpx.AsyncClient(
        transport=ASGITransport(_build_upstream_app()),
        base_url=UPSTREAM_BASE,
    )
    return handler


def _make_agent(*, protocol_version: str) -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "test-agent"
    agent.agent_name = "test-agent"
    agent.agent_card_params = {
        "url": f"{UPSTREAM_BASE}/",
        "name": "Test Agent",
        "protocolVersion": protocol_version,
    }
    agent.litellm_params = {}
    agent.static_headers = None
    agent.extra_headers = None
    return agent


def _make_request(
    method: str,
    params: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    request_id: str = "req-1",
) -> MagicMock:
    request = MagicMock()
    request.headers = headers or {}
    request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
    )
    return request


async def _add_proxy_data(data: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    data["proxy_server_request"] = {
        "url": "http://localhost:4000/a2a/test-agent",
        "method": "POST",
        "headers": {},
        "body": {},
    }
    data.setdefault("metadata", {})
    return data


def _proxy_patches(agent: MagicMock) -> List[Any]:
    from litellm.proxy.agent_endpoints import a2a_endpoints as a2a_endpoints_mod

    return [
        patch.object(a2a_endpoints_mod, "_get_agent", return_value=agent),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler"
            ".AgentRequestHandler.is_agent_allowed",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            new=AsyncMock(side_effect=_add_proxy_data),
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch(
            "litellm.a2a_protocol.main.get_async_httpx_client",
            side_effect=_fake_get_async_httpx_client,
        ),
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
    ]


def _wire_send_params() -> Dict[str, Any]:
    return {
        "message": {
            "role": "user",
            "messageId": "m-in",
            "parts": [{"kind": "text", "text": "ping"}],
        }
    }


def _a2a10_send_params() -> Dict[str, Any]:
    return {
        "message": {
            "role": "ROLE_USER",
            "messageId": "m-in",
            "parts": [{"text": "ping"}],
        },
        "configuration": {},
    }


@pytest.fixture(autouse=True)
def _clear_upstream_calls() -> None:
    _UPSTREAM_CALLS.clear()


@pytest.mark.asyncio
async def test_proxy_serves_1_0_when_agent_pinned_and_upstream_speaks_03():
    from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

    agent = _make_agent(protocol_version="1.0")
    request = _make_request(
        "SendMessage",
        _a2a10_send_params(),
        headers={"a2a-version": "1.0"},
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test", user_id="test-user", team_id="test-team"
    )

    with ExitStack() as stack:
        for item in _proxy_patches(agent):
            stack.enter_context(item)

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert "error" not in body, body
    assert "message" in body["result"]
    assert "kind" not in body["result"]
    assert body["result"]["message"]["parts"][0]["text"] == "pong"
    assert _UPSTREAM_CALLS, "expected upstream to receive a JSON-RPC call"
    assert _UPSTREAM_CALLS[0]["method"] == "message/send"


@pytest.mark.asyncio
async def test_proxy_serves_0_3_when_agent_pinned_passthrough():
    from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

    agent = _make_agent(protocol_version="0.3")
    request = _make_request("message/send", _wire_send_params())
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test", user_id="test-user", team_id="test-team"
    )

    with ExitStack() as stack:
        for item in _proxy_patches(agent):
            stack.enter_context(item)

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert "error" not in body, body
    assert body["result"]["kind"] == "message"
    assert body["result"]["parts"][0]["text"] == "pong"
    assert "message" not in body["result"]


@pytest.mark.asyncio
async def test_proxy_streaming_serves_1_0_envelopes():
    from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

    agent = _make_agent(protocol_version="1.0")
    request = _make_request(
        "SendStreamingMessage",
        _a2a10_send_params(),
        headers={"a2a-version": "1.0"},
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test", user_id="test-user", team_id="test-team"
    )

    with ExitStack() as stack:
        for item in _proxy_patches(agent):
            stack.enter_context(item)

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        lines: List[Dict[str, Any]] = []
        async for raw_line in response.body_iterator:
            line = (
                raw_line.decode().strip()
                if isinstance(raw_line, (bytes, bytearray))
                else str(raw_line).strip()
            )
            if line:
                lines.append(json.loads(line))

    assert lines, "expected at least one streamed JSON-RPC event"
    message_events = [
        line
        for line in lines
        if isinstance(line.get("result"), dict) and "message" in line["result"]
    ]
    assert message_events, f"expected a 1.0 message envelope, got: {lines}"
    assert message_events[-1]["result"]["message"]["parts"][0]["text"] == "pong"
    assert _UPSTREAM_CALLS, "expected upstream streaming call"
    assert _UPSTREAM_CALLS[0]["method"] == "message/stream"
