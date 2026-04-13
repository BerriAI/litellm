"""
Unit tests for A2A agent custom header forwarding.

Tests cover:
- Static headers forwarded to backend agent
- Dynamic headers extracted from client request and forwarded
- Static headers win over dynamic on conflict
- No headers configured — existing behavior unchanged
- merge_agent_headers utility
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal mock agent
# ---------------------------------------------------------------------------

def _make_mock_agent(
    static_headers=None,
    extra_headers=None,
    url="http://backend-agent:10001",
):
    mock_agent = MagicMock()
    mock_agent.agent_id = "agent-123"
    mock_agent.agent_card_params = {"url": url, "name": "Test Agent"}
    mock_agent.litellm_params = {}
    mock_agent.static_headers = static_headers or {}
    mock_agent.extra_headers = extra_headers or []
    return mock_agent


def _make_mock_request(extra_headers=None, method="message/send"):
    """Build a mock FastAPI Request with configurable headers."""
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
                    "messageId": "msg-123",
                }
            },
        }
    )
    return mock_request


def _make_a2a_types_module():
    """Return (module, MessageSendParams, SendMessageRequest, SendStreamingMessageRequest)."""
    try:
        from a2a.types import (
            MessageSendParams,
            SendMessageRequest,
            SendStreamingMessageRequest,
        )
        mock_a2a_types = MagicMock()
        mock_a2a_types.MessageSendParams = MessageSendParams
        mock_a2a_types.SendMessageRequest = SendMessageRequest
        mock_a2a_types.SendStreamingMessageRequest = SendStreamingMessageRequest
        return mock_a2a_types
    except ImportError:
        pass

    def _make_cls(name):
        class MockCls:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self._kwargs = kwargs

            def model_dump(self, mode="json", exclude_none=False):
                result = dict(self._kwargs)
                if exclude_none:
                    result = {k: v for k, v in result.items() if v is not None}
                return result

        MockCls.__name__ = name
        return MockCls

    mock_a2a_types = MagicMock()
    mock_a2a_types.MessageSendParams = _make_cls("MessageSendParams")
    mock_a2a_types.SendMessageRequest = _make_cls("SendMessageRequest")
    mock_a2a_types.SendStreamingMessageRequest = _make_cls(
        "SendStreamingMessageRequest"
    )
    return mock_a2a_types


async def _invoke(mock_agent, mock_request, mock_asend_message):
    """Run invoke_agent_a2a with standard patches applied."""
    from litellm.proxy._types import UserAPIKeyAuth

    mock_user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    mock_fastapi_response = MagicMock()
    mock_a2a_types = _make_a2a_types_module()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"status": "success"},
    }

    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
        return_value=mock_agent,
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
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.proxy_config",
        MagicMock(),
    ), patch(
        "litellm.proxy.proxy_server.version",
        "1.0.0",
    ), patch.dict(
        sys.modules,
        {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
    ), patch(
        "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
        True,
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )
        return mock_asend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_headers_forwarded():
    """Static headers configured on the agent are passed to asend_message."""
    mock_agent = _make_mock_agent(
        static_headers={"Authorization": "Bearer token123"}
    )
    mock_request = _make_mock_request()

    mock_asend = await _invoke(mock_agent, mock_request, None)

    call_kwargs = mock_asend.call_args.kwargs
    headers = call_kwargs.get("agent_extra_headers")
    assert headers is not None, "agent_extra_headers should not be None"
    assert headers.get("Authorization") == "Bearer token123"


@pytest.mark.asyncio
async def test_dynamic_headers_forwarded():
    """Dynamic headers listed in extra_headers are extracted from the client request."""
    mock_agent = _make_mock_agent(extra_headers=["x-api-key"])
    mock_request = _make_mock_request(extra_headers={"x-api-key": "secret"})

    mock_asend = await _invoke(mock_agent, mock_request, None)

    call_kwargs = mock_asend.call_args.kwargs
    headers = call_kwargs.get("agent_extra_headers")
    assert headers is not None
    assert headers.get("x-api-key") == "secret"


@pytest.mark.asyncio
async def test_static_overrides_dynamic():
    """When the same header appears in both static and dynamic, static wins."""
    mock_agent = _make_mock_agent(
        static_headers={"Authorization": "Bearer static-token"},
        extra_headers=["Authorization"],
    )
    # Client sends a different value for Authorization
    mock_request = _make_mock_request(
        extra_headers={"Authorization": "Bearer dynamic-token"}
    )

    mock_asend = await _invoke(mock_agent, mock_request, None)

    call_kwargs = mock_asend.call_args.kwargs
    headers = call_kwargs.get("agent_extra_headers")
    assert headers is not None
    assert headers.get("Authorization") == "Bearer static-token"


@pytest.mark.asyncio
async def test_no_headers():
    """When no headers are configured, agent_extra_headers is None and behaviour is unchanged."""
    mock_agent = _make_mock_agent()  # no static_headers or extra_headers
    mock_request = _make_mock_request()

    mock_asend = await _invoke(mock_agent, mock_request, None)

    call_kwargs = mock_asend.call_args.kwargs
    headers = call_kwargs.get("agent_extra_headers")
    assert headers is None


# ---------------------------------------------------------------------------
# Convention-based x-a2a-{agent_id/name}-{header_name} tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convention_header_by_agent_name():
    """x-a2a-{agent_name}-{header} is forwarded using the agent name alias."""
    mock_agent = _make_mock_agent()
    mock_agent.agent_name = "my-agent"
    mock_request = _make_mock_request(
        extra_headers={"x-a2a-my-agent-authorization": "Bearer conv-token"}
    )

    mock_asend = await _invoke(mock_agent, mock_request, None)

    headers = mock_asend.call_args.kwargs.get("agent_extra_headers")
    assert headers is not None
    assert headers.get("authorization") == "Bearer conv-token"


@pytest.mark.asyncio
async def test_convention_header_by_agent_id():
    """x-a2a-{agent_id}-{header} is forwarded using the agent UUID."""
    mock_agent = _make_mock_agent()
    mock_agent.agent_id = "abc-123"
    mock_agent.agent_name = "other-name"
    mock_request = _make_mock_request(
        extra_headers={"x-a2a-abc-123-x-api-key": "id-secret"}
    )

    mock_asend = await _invoke(mock_agent, mock_request, None)

    headers = mock_asend.call_args.kwargs.get("agent_extra_headers")
    assert headers is not None
    assert headers.get("x-api-key") == "id-secret"


@pytest.mark.asyncio
async def test_convention_header_static_still_wins():
    """Static headers still override convention-based dynamic headers."""
    mock_agent = _make_mock_agent(
        static_headers={"authorization": "Bearer static-wins"}
    )
    mock_agent.agent_name = "my-agent"
    mock_request = _make_mock_request(
        extra_headers={"x-a2a-my-agent-authorization": "Bearer conv-value"}
    )

    mock_asend = await _invoke(mock_agent, mock_request, None)

    headers = mock_asend.call_args.kwargs.get("agent_extra_headers")
    assert headers is not None
    assert headers.get("authorization") == "Bearer static-wins"


@pytest.mark.asyncio
async def test_convention_unrelated_prefix_not_forwarded():
    """Headers that start with x-a2a- but target a different agent are ignored."""
    mock_agent = _make_mock_agent()
    mock_agent.agent_id = "agent-abc"
    mock_agent.agent_name = "my-agent"
    mock_request = _make_mock_request(
        extra_headers={"x-a2a-other-agent-authorization": "Bearer wrong"}
    )

    mock_asend = await _invoke(mock_agent, mock_request, None)

    headers = mock_asend.call_args.kwargs.get("agent_extra_headers")
    assert headers is None


# ---------------------------------------------------------------------------
# Direct unit test for the merge utility
# ---------------------------------------------------------------------------


def test_merge_agent_headers_util_dynamic_only():
    from litellm.proxy.agent_endpoints.utils import merge_agent_headers

    result = merge_agent_headers(dynamic_headers={"x-key": "val"})
    assert result == {"x-key": "val"}


def test_merge_agent_headers_util_static_only():
    from litellm.proxy.agent_endpoints.utils import merge_agent_headers

    result = merge_agent_headers(static_headers={"Authorization": "Bearer tok"})
    assert result == {"Authorization": "Bearer tok"}


def test_merge_agent_headers_util_static_wins():
    from litellm.proxy.agent_endpoints.utils import merge_agent_headers

    result = merge_agent_headers(
        dynamic_headers={"Authorization": "dynamic", "x-extra": "d"},
        static_headers={"Authorization": "static"},
    )
    assert result == {"Authorization": "static", "x-extra": "d"}


def test_merge_agent_headers_util_none_returns_none():
    from litellm.proxy.agent_endpoints.utils import merge_agent_headers

    result = merge_agent_headers()
    assert result is None


def test_merge_agent_headers_util_empty_dicts_returns_none():
    from litellm.proxy.agent_endpoints.utils import merge_agent_headers

    result = merge_agent_headers(dynamic_headers={}, static_headers={})
    assert result is None
