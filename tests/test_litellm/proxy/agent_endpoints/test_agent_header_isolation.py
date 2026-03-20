"""
Tests that prove header isolation between agents.

Before the fix these tests FAIL — agent A's headers bleed into agent B
because create_a2a_client mutates a globally cached httpx client.
After the fix they pass.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(agent_id, agent_name, static_headers=None, extra_headers=None, url="http://0.0.0.0:9999"):
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
        from a2a.types import MessageSendParams, SendMessageRequest, SendStreamingMessageRequest
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
    mock_response.model_dump.return_value = {"jsonrpc": "2.0", "id": "test-id", "result": {}}

    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
        return_value=agent,
    ), patch(
        "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
        new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "litellm.proxy.common_request_processing.add_litellm_data_to_request",
        side_effect=lambda data, **kw: data,
    ), patch(
        "litellm.a2a_protocol.asend_message",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_asend, patch(
        "litellm.a2a_protocol.create_a2a_client",
        new_callable=AsyncMock,
    ), patch(
        "litellm.proxy.proxy_server.general_settings", {}
    ), patch(
        "litellm.proxy.proxy_server.proxy_config", MagicMock()
    ), patch(
        "litellm.proxy.proxy_server.version", "1.0.0"
    ), patch.dict(
        sys.modules,
        {"a2a": MagicMock(), "a2a.types": _a2a_types_module()},
    ), patch(
        "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True
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
    agent_a = _make_agent("id-a", "agent-a", static_headers={"X-Agent-A-Token": "secret-a"})
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
# Unit test: create_a2a_client uses a fresh httpx client per call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_a2a_client_uses_fresh_httpx_client():
    """
    Two calls to create_a2a_client with different extra_headers must NOT
    share the same underlying httpx.AsyncClient instance.
    """
    import httpx

    from litellm.a2a_protocol.main import create_a2a_client

    created_clients = []

    fake_agent_card = MagicMock()
    fake_agent_card.name = "test-agent"

    class FakeResolver:
        def __init__(self, **kw):
            created_clients.append(kw.get("httpx_client"))
        async def get_agent_card(self):
            return fake_agent_card

    class FakeA2AClient:
        def __init__(self, httpx_client, agent_card):
            self._client = httpx_client
            self._litellm_agent_card = agent_card

    with patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True), patch(
        "litellm.a2a_protocol.main.A2ACardResolver", FakeResolver
    ), patch("litellm.a2a_protocol.main._A2AClient", FakeA2AClient):
        await create_a2a_client(
            base_url="http://agent-a:9999",
            extra_headers={"Authorization": "Bearer a"},
        )
        await create_a2a_client(
            base_url="http://agent-b:9999",
            extra_headers={"Authorization": "Bearer b"},
        )

    assert len(created_clients) == 2
    # Must be distinct objects
    assert created_clients[0] is not created_clients[1], (
        "create_a2a_client reused a cached httpx client — headers will bleed between agents"
    )
