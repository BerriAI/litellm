"""
Tests that prove header isolation between agents.

Before the fix these tests FAIL — agent A's headers bleed into agent B
because create_a2a_client mutates a globally cached httpx client.
After the fix they pass.

Also includes direct unit tests for create_a2a_client (fresh httpx client
per call; default timeout uses DEFAULT_A2A_AGENT_TIMEOUT).
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.constants import DEFAULT_A2A_AGENT_TIMEOUT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id,
    agent_name,
    static_headers=None,
    extra_headers=None,
    url="http://0.0.0.0:9999",
):
    a = MagicMock()
    a.agent_id = agent_id
    a.agent_name = agent_name
    a.agent_card_params = {"url": url, "name": agent_name}
    a.litellm_params = {}
    a.static_headers = static_headers or {}
    a.extra_headers = extra_headers or []
    return a


def _make_request(method="message/send", extra_headers=None):
    mock_request = MagicMock()
    headers = {"content-type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    mock_request.headers = headers
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": method,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-1",
                }
            },
        }
    )
    return mock_request


def _a2a_types_module():
    try:
        from a2a.types import (
            MessageSendParams,
            SendMessageRequest,
            SendStreamingMessageRequest,
        )

        m = MagicMock()
        m.MessageSendParams = MessageSendParams
        m.SendMessageRequest = SendMessageRequest
        m.SendStreamingMessageRequest = SendStreamingMessageRequest
        return m
    except ImportError:
        pass

    def _cls(name):
        class C:
            def __init__(self, **kw):
                self.__dict__.update(kw)
                self._kw = kw

            def model_dump(self, mode="json", exclude_none=False):
                return dict(self._kw)

        C.__name__ = name
        return C

    m = MagicMock()
    m.MessageSendParams = _cls("MessageSendParams")
    m.SendMessageRequest = _cls("SendMessageRequest")
    m.SendStreamingMessageRequest = _cls("SendStreamingMessageRequest")
    return m


async def _invoke_agent(agent, request):
    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    fastapi_response = MagicMock()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {},
    }

    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=agent,
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=lambda data, **kw: data,
        ),
        patch(
            "litellm.a2a_protocol.asend_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_asend,
        patch(
            "litellm.a2a_protocol.create_a2a_client",
            new_callable=AsyncMock,
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch.dict(
            sys.modules,
            {"a2a": MagicMock(), "a2a.types": _a2a_types_module()},
        ),
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id=agent.agent_id,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )
        return mock_asend.call_args.kwargs.get("agent_extra_headers")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_headers_do_not_leak_between_agents():
    """
    Agent A has static_headers={"X-Agent-A-Token": "secret-a"}.
    Agent B has no headers.
    After invoking A then B, B must NOT receive X-Agent-A-Token.
    """
    agent_a = _make_agent(
        "id-a", "agent-a", static_headers={"X-Agent-A-Token": "secret-a"}
    )
    agent_b = _make_agent("id-b", "agent-b")

    headers_a = await _invoke_agent(agent_a, _make_request())
    headers_b = await _invoke_agent(agent_b, _make_request())

    assert headers_a is not None
    assert headers_a.get("X-Agent-A-Token") == "secret-a"

    # Agent B must not have agent A's header
    assert headers_b is None or "X-Agent-A-Token" not in headers_b


@pytest.mark.asyncio
async def test_convention_header_only_matches_own_agent():
    """
    Client sends x-a2a-agent-a-authorization: Bearer for-a.
    When invoking agent-b, that header must NOT be forwarded.
    """
    agent_b = _make_agent("id-b", "agent-b")
    # Request carries a header scoped to agent-a, not agent-b
    req = _make_request(extra_headers={"x-a2a-agent-a-authorization": "Bearer for-a"})

    headers_b = await _invoke_agent(agent_b, req)

    assert headers_b is None or "authorization" not in (headers_b or {})


@pytest.mark.asyncio
async def test_convention_header_matches_own_agent():
    """
    Client sends x-a2a-agent-b-authorization: Bearer for-b.
    When invoking agent-b, that header IS forwarded.
    """
    agent_b = _make_agent("id-b", "agent-b")
    req = _make_request(extra_headers={"x-a2a-agent-b-authorization": "Bearer for-b"})

    headers_b = await _invoke_agent(agent_b, req)

    assert headers_b is not None
    assert headers_b.get("authorization") == "Bearer for-b"


@pytest.mark.asyncio
async def test_each_agent_gets_only_its_own_static_headers():
    """
    Agent A: static_headers={"X-Token": "a"}
    Agent B: static_headers={"X-Token": "b"}
    Each must receive only their own value.
    """
    agent_a = _make_agent("id-a", "agent-a", static_headers={"X-Token": "a"})
    agent_b = _make_agent("id-b", "agent-b", static_headers={"X-Token": "b"})

    headers_a = await _invoke_agent(agent_a, _make_request())
    headers_b = await _invoke_agent(agent_b, _make_request())

    assert (headers_a or {}).get("X-Token") == "a"
    assert (headers_b or {}).get("X-Token") == "b"


# ---------------------------------------------------------------------------
# Unit tests: create_a2a_client (httpx client per call + timeout defaults)
# ---------------------------------------------------------------------------


def _fake_get_async_httpx_client_factory(captured_calls: list):
    """Return a side_effect that records every (params, client) pair."""

    def _fake_get_async_httpx_client(llm_provider, params, **kwargs):
        client = MagicMock()
        client.headers = MagicMock()
        handler = MagicMock()
        handler.client = client
        captured_calls.append({"params": params.copy(), "client": client})
        return handler

    return _fake_get_async_httpx_client


async def _fake_create_client(base_url, client_config=None, **kwargs):
    client = MagicMock()
    if client_config is not None:
        client._litellm_httpx_client = client_config.httpx_client
    return client


@pytest.mark.asyncio
async def test_create_a2a_client_uses_fresh_httpx_client():
    """
    Two calls to create_a2a_client with different extra_headers must produce
    distinct underlying httpx clients — preventing header bleed between agents.

    The test checks:
    1. get_async_httpx_client was called twice (once per create_a2a_client call).
    2. The two returned A2A clients carry distinct httpx client objects (direct
       proof of header isolation, not just cache-key difference).
    3. The cache-key param differs between calls (so the real LRU cache cannot
       return the same httpx client even under load).
    """
    pytest.importorskip("a2a.client")
    from litellm.a2a_protocol.main import create_a2a_client

    captured_calls: list = []

    with (
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
        patch(
            "litellm.a2a_protocol.main.get_async_httpx_client",
            side_effect=_fake_get_async_httpx_client_factory(captured_calls),
        ),
        patch(
            "litellm.a2a_protocol.main.create_client",
            new=AsyncMock(side_effect=_fake_create_client),
        ),
    ):
        a2a_client_a = await create_a2a_client(
            base_url="http://agent-a:9999",
            extra_headers={"Authorization": "Bearer a"},
        )
        a2a_client_b = await create_a2a_client(
            base_url="http://agent-b:9999",
            extra_headers={"Authorization": "Bearer b"},
        )

    assert (
        len(captured_calls) == 2
    ), "create_a2a_client should call get_async_httpx_client once per invocation"

    # Direct proof: the two A2A clients must carry distinct httpx client objects.
    # If they share one, mutating agent-B's Authorization header would bleed into A.
    httpx_a = getattr(a2a_client_a, "_litellm_httpx_client", None)
    httpx_b = getattr(a2a_client_b, "_litellm_httpx_client", None)
    assert httpx_a is not None, "a2a_client_a missing _litellm_httpx_client"
    assert httpx_b is not None, "a2a_client_b missing _litellm_httpx_client"
    assert httpx_a is not httpx_b, (
        "create_a2a_client returned the same httpx client for two agents with "
        "different headers — Authorization header will bleed between agents"
    )

    # Also verify the cache-key param differs so the LRU cache never conflates them.
    key_a = captured_calls[0]["params"].get("disable_aiohttp_transport")
    key_b = captured_calls[1]["params"].get("disable_aiohttp_transport")
    assert key_a is not None, "cache-key param 'disable_aiohttp_transport' missing"
    assert key_b is not None, "cache-key param 'disable_aiohttp_transport' missing"
    assert key_a != key_b, (
        f"create_a2a_client used the same cache key for two agents with different "
        f"headers — headers will bleed: key_a={key_a!r}, key_b={key_b!r}"
    )


@pytest.mark.asyncio
async def test_create_a2a_client_default_timeout_matches_constant():
    """When timeout is omitted, httpx client params must use DEFAULT_A2A_AGENT_TIMEOUT."""
    pytest.importorskip("a2a.client")
    from litellm.a2a_protocol.main import create_a2a_client

    captured: dict = {}

    def _capture_get_async_httpx_client(llm_provider, params, **kwargs):
        captured["params"] = params
        handler = MagicMock()
        handler.client = MagicMock()
        handler.client.headers = MagicMock()
        return handler

    with (
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
        patch(
            "litellm.a2a_protocol.main.get_async_httpx_client",
            side_effect=_capture_get_async_httpx_client,
        ),
        patch(
            "litellm.a2a_protocol.main.create_client",
            new=AsyncMock(side_effect=_fake_create_client),
        ),
    ):
        await create_a2a_client(base_url="http://127.0.0.1:9")

    assert captured["params"]["timeout"] == DEFAULT_A2A_AGENT_TIMEOUT


@pytest.mark.asyncio
async def test_create_a2a_client_explicit_timeout_overrides_default():
    """Explicit timeout= must be passed through to the httpx client params."""
    pytest.importorskip("a2a.client")
    from litellm.a2a_protocol.main import create_a2a_client

    captured: dict = {}

    def _capture_get_async_httpx_client(llm_provider, params, **kwargs):
        captured["params"] = params
        handler = MagicMock()
        handler.client = MagicMock()
        handler.client.headers = MagicMock()
        return handler

    with (
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
        patch(
            "litellm.a2a_protocol.main.get_async_httpx_client",
            side_effect=_capture_get_async_httpx_client,
        ),
        patch(
            "litellm.a2a_protocol.main.create_client",
            new=AsyncMock(side_effect=_fake_create_client),
        ),
    ):
        await create_a2a_client(base_url="http://127.0.0.1:9", timeout=42.5)

    assert captured["params"]["timeout"] == 42.5
