"""
Mock tests for A2A endpoints.

Tests that invoke_agent_a2a properly integrates with add_litellm_data_to_request.
"""

import json
import sys
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_invoke_agent_a2a_adds_litellm_data():
    """
    Test that invoke_agent_a2a calls add_litellm_data_to_request
    and the resulting data includes proxy_server_request.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    # Track the data passed to add_litellm_data_to_request
    captured_data = {}

    async def mock_add_litellm_data(data, **kwargs):
        # Simulate what add_litellm_data_to_request does
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/a2a/test-agent",
            "method": "POST",
            "headers": {},
            "body": dict(data),
        }
        captured_data.update(data)
        return data

    # Mock response from asend_message
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"status": "success"},
    }

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.agent_card_params = {
        "url": "http://backend-agent:10001",
        "name": "Test Agent",
    }
    mock_agent.litellm_params = None

    # Mock request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-123",
                }
            },
        }
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="test-user",
        team_id="test-team",
    )

    # Try to use real a2a.types if available, otherwise create realistic mocks
    # This test focuses on LiteLLM integration, not A2A protocol correctness,
    # but we want mocks that behave like the real types to catch usage issues
    try:
        from a2a.types import (
            MessageSendParams,
            SendMessageRequest,
            SendStreamingMessageRequest,
        )

        # Real types available - use them
        pass
    except ImportError:
        # Real types not available - create realistic mocks
        pass

        def make_mock_pydantic_class(name):
            """Create a mock class that behaves like a Pydantic model."""

            class MockPydanticClass:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    # Store kwargs for model_dump() if needed
                    self._kwargs = kwargs

                def model_dump(self, mode="json", exclude_none=False):
                    """Mock model_dump method."""
                    result = dict(self._kwargs)
                    if exclude_none:
                        result = {k: v for k, v in result.items() if v is not None}
                    return result

            MockPydanticClass.__name__ = name
            return MockPydanticClass

        MessageSendParams = make_mock_pydantic_class("MessageSendParams")
        SendMessageRequest = make_mock_pydantic_class("SendMessageRequest")
        SendStreamingMessageRequest = make_mock_pydantic_class(
            "SendStreamingMessageRequest"
        )

    # Create a mock module for a2a.types
    mock_a2a_types = MagicMock()
    mock_a2a_types.MessageSendParams = MessageSendParams
    mock_a2a_types.SendMessageRequest = SendMessageRequest
    mock_a2a_types.SendStreamingMessageRequest = SendStreamingMessageRequest

    # Patch at the source modules
    # Note: add_litellm_data_to_request is called from common_request_processing,
    # so we need to patch it there, not at litellm_pre_call_utils
    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=mock_agent,
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ) as mock_add_data,
        patch(
            "litellm.a2a_protocol.create_a2a_client",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.a2a_protocol.asend_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.proxy_server.version",
            "1.0.0",
        ),
        patch.dict(
            sys.modules,
            {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
        ),
        patch(
            "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
            True,
        ),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        mock_fastapi_response = MagicMock()

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify add_litellm_data_to_request was called
        mock_add_data.assert_called_once()

        # Verify model and custom_llm_provider were set
        assert captured_data.get("model") == "a2a_agent/Test Agent"
        assert captured_data.get("custom_llm_provider") == "a2a_agent"

        # Verify proxy_server_request was added
        assert "proxy_server_request" in captured_data
        assert captured_data["proxy_server_request"]["method"] == "POST"


@pytest.mark.asyncio
async def test_invoke_agent_a2a_handles_none_agent_card_params():
    """Agents without ``agent_card_params`` (e.g. plain chat agents routed
    through the A2A endpoint by mistake) must not raise ``AttributeError`` on
    ``agent_card_params.get(...)`` — they should return a JSON-RPC error.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    mock_agent = MagicMock()
    mock_agent.agent_card_params = None
    mock_agent.litellm_params = None

    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-123",
                }
            },
        }
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="test-user",
        team_id="test-team",
    )

    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=mock_agent,
        ),
        patch(
            "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
            True,
        ),
        patch.dict(sys.modules, {"a2a": MagicMock(), "a2a.types": MagicMock()}),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        mock_fastapi_response = MagicMock()

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # JSONResponse exposes the body bytes; decode and verify it's a
        # JSON-RPC error, not an "internal error" from a Python exception.
        body = json.loads(response.body.decode())
        assert body["jsonrpc"] == "2.0"
        assert body["error"]["code"] == -32000
        assert "no URL configured" in body["error"]["message"]


@pytest.mark.asyncio
async def test_invoke_agent_a2a_injects_authenticated_key_hash_for_bridge():
    """Completion-bridge agents must receive the authenticated key hash in
    litellm_params so provider configs (e.g. LangFlow) can scope provider-side
    session memory per key. Regression for cross-key A2A session bleed."""
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2A_USER_API_KEY_HASH_PARAM,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    captured = {}

    async def mock_add_litellm_data(data, **kwargs):
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/a2a/lf-agent",
            "method": "POST",
            "headers": {},
            "body": {},
        }
        data.setdefault("metadata", {})
        return data

    async def capture_asend_message(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.model_dump.return_value = {"jsonrpc": "2.0", "id": "test-id", "result": {}}
        return resp

    mock_agent = MagicMock()
    mock_agent.agent_id = "lf-agent"
    mock_agent.agent_name = "lf-agent"
    # No URL: the bridge derives the endpoint from the LangFlow agent config.
    mock_agent.agent_card_params = {"name": "LF Agent"}
    mock_agent.litellm_params = {
        "custom_llm_provider": "langflow",
        "model": "langflow/flow-1",
    }
    mock_agent.static_headers = None
    mock_agent.extra_headers = None

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "msg-1",
                    "contextId": "ctx-1",
                }
            },
        }
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-hashed-123",
        user_id="test-user",
        team_id="test-team",
    )

    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=mock_agent,
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.asend_message",
            new=AsyncMock(side_effect=capture_asend_message),
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True),
        patch.dict(sys.modules, {"a2a": MagicMock(), "a2a.types": MagicMock()}),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id="lf-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=mock_user_api_key_dict,
        )

    assert (
        captured.get("litellm_params", {}).get(A2A_USER_API_KEY_HASH_PARAM)
        == mock_user_api_key_dict.api_key
    ), "authenticated key hash was not forwarded to the completion bridge"


def _make_agent_mock(url: str = "http://backend-agent:10001") -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "test-agent"
    agent.agent_name = "test-agent"
    agent.agent_card_params = {"url": url, "name": "Test Agent"}
    agent.litellm_params = {}
    agent.static_headers = None
    agent.extra_headers = None
    return agent


def _make_request_mock(method: str, params: dict) -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": method,
            "params": params,
        }
    )
    return req


def _base_patches(agent: MagicMock):
    return [
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=agent,
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            new=AsyncMock(side_effect=_add_proxy_data),
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
    ]


async def _add_proxy_data(data, **kwargs):
    data["proxy_server_request"] = {"url": "http://localhost:4000", "method": "POST", "headers": {}, "body": {}}
    data.setdefault("metadata", {})
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,params",
    [
        ("tasks/get", {"id": "task-1"}),
        ("tasks/list", {"contextId": "ctx-1"}),
        ("tasks/cancel", {"id": "task-1"}),
        ("tasks/pushNotificationConfig/set", {"taskId": "task-1", "url": "https://webhook.example.com"}),
        ("tasks/pushNotificationConfig/get", {"taskId": "task-1", "id": "cfg-1"}),
        ("tasks/pushNotificationConfig/list", {"taskId": "task-1"}),
        ("tasks/pushNotificationConfig/delete", {"taskId": "task-1", "id": "cfg-1"}),
    ],
)
async def test_task_methods_forward_jsonrpc(method: str, params: dict):
    from litellm.proxy._types import UserAPIKeyAuth

    upstream_response = {"jsonrpc": "2.0", "id": "req-1", "result": {"id": "task-1", "status": {"state": "completed"}}}
    agent = _make_agent_mock()
    mock_request = _make_request_mock(method, params)

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_http_response)

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(patch("litellm.proxy.agent_endpoints.a2a_endpoints.httpx.AsyncClient", return_value=mock_http_client))

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert body["jsonrpc"] == "2.0"
    assert body["result"]["id"] == "task-1"

    posted = mock_http_client.post.call_args
    assert posted is not None
    forwarded_body = posted.kwargs.get("json") or posted.args[1]
    assert forwarded_body["method"] == method


@pytest.mark.asyncio
async def test_subscribe_to_task_returns_sse_stream():
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("SubscribeToTask", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    sse_lines = [
        'data: {"jsonrpc":"2.0","id":"req-1","result":{"taskId":"task-1","status":{"state":"working"}}}',
        'data: {"jsonrpc":"2.0","id":"req-1","result":{"taskId":"task-1","status":{"state":"completed"}}}',
    ]

    async def fake_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_response = MagicMock()
    mock_stream_response.raise_for_status = MagicMock()
    mock_stream_response.aiter_lines = fake_aiter_lines
    mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_response.__aexit__ = AsyncMock(return_value=False)

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.stream = MagicMock(return_value=mock_stream_response)

    chunks = []
    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(patch("litellm.proxy.agent_endpoints.a2a_endpoints.httpx.AsyncClient", return_value=mock_http_client))

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        assert response.media_type == "text/event-stream"
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    full = "".join(chunks)
    assert "working" in full
    assert "completed" in full


@pytest.mark.asyncio
async def test_get_extended_agent_card_rewrites_url():
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("GetExtendedAgentCard", {})
    mock_request.base_url = "http://localhost:4000/"
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    upstream_card = {
        "name": "Test Agent",
        "url": "http://backend-agent:10001",
        "description": "A test agent",
    }
    upstream_response = {"jsonrpc": "2.0", "id": "req-1", "result": upstream_card}

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_http_response)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(patch("litellm.proxy.agent_endpoints.a2a_endpoints.httpx.AsyncClient", return_value=mock_http_client))

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert body["result"]["url"] == "http://localhost:4000/a2a/test-agent"
    assert body["result"]["name"] == "Test Agent"


@pytest.mark.asyncio
async def test_unknown_method_returns_jsonrpc_error():
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("SomeUnknownMethod", {})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert body["error"]["code"] == -32601
    assert "SomeUnknownMethod" in body["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pascal_method,expected_wire_method",
    [
        ("GetTask", "tasks/get"),
        ("ListTasks", "tasks/list"),
        ("CancelTask", "tasks/cancel"),
        ("SubscribeToTask", "tasks/resubscribe"),
        ("CreateTaskPushNotificationConfig", "tasks/pushNotificationConfig/set"),
        ("GetTaskPushNotificationConfig", "tasks/pushNotificationConfig/get"),
        ("ListTaskPushNotificationConfigs", "tasks/pushNotificationConfig/list"),
        ("DeleteTaskPushNotificationConfig", "tasks/pushNotificationConfig/delete"),
        ("GetExtendedAgentCard", "agent/getAuthenticatedExtendedCard"),
    ],
)
async def test_pascal_method_names_normalize_to_wire_format(pascal_method: str, expected_wire_method: str):
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(pascal_method, {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    upstream_response = {"jsonrpc": "2.0", "id": "req-1", "result": {"id": "task-1"}}
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_http_response)

    async def _empty_aiter_lines():
        return
        yield  # make it an async generator

    sse_response = MagicMock()
    sse_response.raise_for_status = MagicMock()
    sse_response.aiter_lines = _empty_aiter_lines
    sse_response.__aenter__ = AsyncMock(return_value=sse_response)
    sse_response.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.stream = MagicMock(return_value=sse_response)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(patch("litellm.proxy.agent_endpoints.a2a_endpoints.httpx.AsyncClient", return_value=mock_http_client))

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        if expected_wire_method == "tasks/resubscribe":
            assert response.media_type == "text/event-stream"
            async for _ in response.body_iterator:
                pass
        else:
            body = json.loads(response.body.decode())
            assert "error" not in body, f"Got error: {body}"

    if expected_wire_method != "tasks/resubscribe":
        posted = mock_http_client.post.call_args
        forwarded_body = posted.kwargs.get("json") or posted.args[1]
        assert forwarded_body["method"] == expected_wire_method, (
            f"Expected '{expected_wire_method}' forwarded for PascalCase '{pascal_method}', "
            f"but got '{forwarded_body['method']}'"
        )
