"""
Mock tests for A2A endpoints.

Tests that invoke_agent_a2a properly integrates with add_litellm_data_to_request.
"""

import json
import socket
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


def _make_request_mock(
    method: str, params: dict, request_id: object = "req-1"
) -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": request_id,
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
    data["proxy_server_request"] = {
        "url": "http://localhost:4000",
        "method": "POST",
        "headers": {},
        "body": {},
    }
    data.setdefault("metadata", {})
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["message/send", "message/stream"])
async def test_message_methods_preserve_numeric_zero_request_id(method: str):
    from fastapi.responses import JSONResponse
    from litellm.proxy._types import UserAPIKeyAuth

    class MessageSendParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SendMessageRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    agent = _make_agent_mock()
    params = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello"}],
            "messageId": "msg-123",
        }
    }
    mock_request = _make_request_mock(method, params, request_id=0)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")
    captured = {}

    async def capture_asend_message(request, **kwargs):
        captured["request_id"] = request.id
        response = MagicMock()
        response.model_dump.return_value = {
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {"status": "success"},
        }
        return response

    async def capture_stream_message(**kwargs):
        captured["request_id"] = kwargs["request_id"]
        return JSONResponse({"jsonrpc": "2.0", "id": kwargs["request_id"]})

    mock_a2a_types = MagicMock()
    mock_a2a_types.MessageSendParams = MessageSendParams
    mock_a2a_types.SendMessageRequest = SendMessageRequest

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True))
        if method == "message/send":
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
                )
            )
            stack.enter_context(
                patch(
                    "litellm.a2a_protocol.asend_message",
                    new=AsyncMock(side_effect=capture_asend_message),
                )
            )
        else:
            stack.enter_context(
                patch(
                    "litellm.proxy.agent_endpoints.a2a_endpoints._handle_stream_message",
                    new=AsyncMock(side_effect=capture_stream_message),
                )
            )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    assert captured["request_id"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,params",
    [
        ("tasks/get", {"id": "task-1"}),
        ("tasks/list", {"contextId": "ctx-1"}),
        ("tasks/cancel", {"id": "task-1"}),
        (
            "tasks/pushNotificationConfig/set",
            {"taskId": "task-1", "url": "https://webhook.example.com"},
        ),
        ("tasks/pushNotificationConfig/get", {"taskId": "task-1", "id": "cfg-1"}),
        ("tasks/pushNotificationConfig/list", {"taskId": "task-1"}),
        ("tasks/pushNotificationConfig/delete", {"taskId": "task-1", "id": "cfg-1"}),
    ],
)
async def test_task_methods_forward_jsonrpc(method: str, params: dict):
    from litellm.proxy._types import UserAPIKeyAuth

    upstream_response = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "task-1", "status": {"state": "completed"}},
    }
    agent = _make_agent_mock()
    mock_request = _make_request_mock(method, params)

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True
    mock_http_response.raise_for_status = MagicMock()

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = MagicMock()

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.agent_endpoints.a2a_endpoints.validate_url",
                return_value=("https://webhook.example.com", "webhook.example.com"),
            )
        )

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

    posted = mock_handler.post.call_args
    assert posted is not None
    forwarded_body = posted.kwargs.get("json") or posted.args[1]
    assert forwarded_body["method"] == method


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["tasks/get", "tasks/resubscribe"])
async def test_task_methods_extract_litellm_params_before_forwarding(method: str):
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    params = {
        "id": "task-1",
        "guardrails": ["guardrail-1"],
        "tags": ["tag-1"],
    }
    mock_request = _make_request_mock(method, params)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")
    captured_data = {}

    async def capture_proxy_data(data, **kwargs):
        captured_data.update(data)
        return await _add_proxy_data(data, **kwargs)

    upstream_response = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "task-1", "status": {"state": "completed"}},
    }
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True

    async def fake_aiter_lines():
        yield 'data: {"jsonrpc":"2.0","id":"req-1","result":{"taskId":"task-1"}}'

    mock_resp = AsyncMock()
    mock_resp.is_success = True
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = mock_async_client

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.proxy.common_request_processing.add_litellm_data_to_request",
                new=AsyncMock(side_effect=capture_proxy_data),
            )
        )
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )
        if method == "tasks/resubscribe":
            async for _ in response.body_iterator:
                pass

    if method == "tasks/resubscribe":
        forwarded_body = mock_async_client.build_request.call_args.kwargs["json"]
    else:
        forwarded_body = mock_handler.post.call_args.kwargs["json"]
    assert forwarded_body["params"] == {"id": "task-1"}
    assert captured_data["guardrails"] == ["guardrail-1"]
    assert captured_data["tags"] == ["tag-1"]


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

    mock_resp = AsyncMock()
    mock_resp.is_success = True
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.client = mock_async_client
    mock_handler.post = AsyncMock()

    chunks = []
    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

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
async def test_subscribe_to_task_calls_pre_call_hook():
    """tasks/resubscribe must run pre_call_hook so guardrails configured on
    the agent are enforced before streaming begins."""
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/resubscribe", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    async def fake_aiter_lines():
        yield 'data: {"jsonrpc":"2.0","id":"req-1","result":{"taskId":"task-1","status":{"state":"completed"}}}'

    mock_resp = AsyncMock()
    mock_resp.is_success = True
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.client = mock_async_client
    mock_handler.post = AsyncMock()

    async def _passthrough_iterator(response, **kwargs):
        async for chunk in response:
            yield chunk

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(
        side_effect=lambda user_api_key_dict, data, call_type: data
    )
    mock_proxy_logging.async_post_call_streaming_iterator_hook = _passthrough_iterator
    mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj",
                mock_proxy_logging,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        assert response.media_type == "text/event-stream"
        async for _ in response.body_iterator:
            pass

    mock_proxy_logging.pre_call_hook.assert_awaited_once()
    call_kwargs = mock_proxy_logging.pre_call_hook.await_args.kwargs
    assert call_kwargs.get("call_type") == "asend_message"
    assert call_kwargs.get("user_api_key_dict") == user_api_key_dict


@pytest.mark.asyncio
async def test_subscribe_to_task_runs_post_call_streaming_guardrail():
    """tasks/resubscribe must route streamed events through the post-call
    streaming hook so output guardrails configured on the agent inspect the
    streamed task content. Regression: the SSE path previously returned the raw
    upstream stream and bypassed guardrails entirely."""
    import litellm
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import UserAPIKeyAuth

    inspected: list = []

    class _RecordingGuardrail(CustomGuardrail):
        async def async_post_call_streaming_hook(self, user_api_key_dict, response):
            inspected.append(response)
            return response

    guardrail = _RecordingGuardrail(
        guardrail_name="record-a2a", default_on=True, event_hook="post_call"
    )

    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/resubscribe", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    async def fake_aiter_lines():
        yield (
            'data: {"jsonrpc":"2.0","id":"req-1","result":'
            '{"kind":"message","parts":[{"kind":"text","text":"resubscribe-secret"}]}}'
        )

    mock_resp = AsyncMock()
    mock_resp.is_success = True
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.client = mock_async_client
    mock_handler.post = AsyncMock()

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )
        stack.enter_context(patch.object(litellm, "callbacks", [guardrail]))

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        assert response.media_type == "text/event-stream"
        async for _ in response.body_iterator:
            pass

    assert any("resubscribe-secret" in str(r) for r in inspected), (
        "tasks/resubscribe streamed content was not passed to the post-call "
        "streaming guardrail hook"
    )


@pytest.mark.asyncio
async def test_task_method_failure_hook_uses_enriched_request_data():
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/get", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    async def add_proxy_data_copy(data, **kwargs):
        enriched = dict(data)
        enriched["proxy_server_request"] = {
            "url": "http://localhost:4000",
            "method": "POST",
            "headers": {},
            "body": {},
        }
        enriched.setdefault("metadata", {})
        return enriched

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(side_effect=RuntimeError("upstream failed"))

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(
        side_effect=lambda user_api_key_dict, data, call_type: data
    )
    mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.proxy.common_request_processing.add_litellm_data_to_request",
                new=AsyncMock(side_effect=add_proxy_data_copy),
            )
        )
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj",
                mock_proxy_logging,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert body["error"]["code"] == -32603
    failure_data = mock_proxy_logging.post_call_failure_hook.await_args.kwargs[
        "request_data"
    ]
    assert failure_data.get("litellm_call_id")
    assert failure_data.get("agent_id") == "test-agent"


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
    mock_http_response.is_success = True
    mock_http_response.raise_for_status = MagicMock()

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = MagicMock()

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

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
async def test_pascal_method_names_normalize_to_wire_format(
    pascal_method: str, expected_wire_method: str
):
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(pascal_method, {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    upstream_response = {"jsonrpc": "2.0", "id": "req-1", "result": {"id": "task-1"}}
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True
    mock_http_response.raise_for_status = MagicMock()

    async def _empty_aiter_lines():
        return
        yield  # make it an async generator

    mock_sse_resp = AsyncMock()
    mock_sse_resp.is_success = True
    mock_sse_resp.aiter_lines = _empty_aiter_lines
    mock_sse_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_sse_resp)

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = mock_async_client

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

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
        posted = mock_handler.post.call_args
        forwarded_body = posted.kwargs.get("json") or posted.args[1]
        assert forwarded_body["method"] == expected_wire_method, (
            f"Expected '{expected_wire_method}' forwarded for PascalCase '{pascal_method}', "
            f"but got '{forwarded_body['method']}'"
        )


@pytest.mark.asyncio
async def test_task_method_upstream_jsonrpc_error_on_http_4xx_is_relayed():
    """When upstream returns HTTP 4xx with a JSON-RPC error body, the error body
    must be relayed to the client unchanged, not replaced with a generic string."""
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/get", {"id": "nonexistent"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    upstream_error = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "error": {"code": -32001, "message": "Task not found"},
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_error
    mock_http_response.is_success = False
    mock_http_response.raise_for_status = MagicMock(
        side_effect=Exception("404 Not Found")
    )

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = MagicMock()

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    body = json.loads(response.body.decode())
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Task not found"


@pytest.mark.asyncio
async def test_subscribe_to_task_upstream_error_yields_jsonrpc_error_event():
    """When upstream returns a non-2xx response for tasks/resubscribe, the SSE
    stream must yield a JSON-RPC error event instead of silently breaking."""
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/resubscribe", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    mock_resp = AsyncMock()
    mock_resp.is_success = False
    mock_resp.status_code = 404
    mock_resp.reason_phrase = "Not Found"
    mock_resp.aread = AsyncMock(
        return_value=b'{"jsonrpc":"2.0","error":{"code":-32001,"message":"Task not found"}}'
    )
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.client = mock_async_client
    mock_handler.post = AsyncMock()

    chunks = []
    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

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
    body = json.loads(full.removeprefix("data: ").strip())
    assert body["id"] == "req-1"
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Task not found"


@pytest.mark.asyncio
async def test_forward_jsonrpc_sse_fallback_error_uses_jsonrpc_error_code():
    mock_resp = AsyncMock()
    mock_resp.is_success = False
    mock_resp.status_code = 503
    mock_resp.reason_phrase = "Service Unavailable"
    mock_resp.aread = AsyncMock(return_value=b"upstream unavailable")
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.client = mock_async_client

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_handler,
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import _forward_jsonrpc_sse

        response = await _forward_jsonrpc_sse(
            agent_url="http://backend-agent:10001",
            body={"jsonrpc": "2.0", "id": "req-1", "method": "tasks/resubscribe"},
            request_id="req-1",
        )

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    body = json.loads("".join(chunks).removeprefix("data: ").strip())
    assert body["error"]["code"] == -32603
    assert body["error"]["message"] == "Service Unavailable"


@pytest.mark.asyncio
async def test_task_methods_forward_caller_identity_headers():
    """Task operations must forward X-LiteLLM-User-Id and X-LiteLLM-Team-Id so the
    upstream agent can scope resources to the authenticated caller."""
    from litellm.proxy._types import UserAPIKeyAuth

    upstream_response = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "task-1", "status": {"state": "completed"}},
    }
    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/get", {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test", user_id="user-abc", team_id="team-xyz"
    )

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    posted_headers = mock_handler.post.call_args.kwargs.get("headers") or {}
    assert posted_headers.get("X-LiteLLM-User-Id") == "user-abc"
    assert posted_headers.get("X-LiteLLM-Team-Id") == "team-xyz"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["tasks/get", "tasks/resubscribe"])
async def test_task_methods_forward_trace_header(method: str):
    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(method, {"id": "task-1"})
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    async def add_proxy_data_with_trace(data, **kwargs):
        data = await _add_proxy_data(data, **kwargs)
        data["litellm_trace_id"] = "trace-123"
        return data

    upstream_response = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "task-1", "status": {"state": "completed"}},
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True

    async def fake_aiter_lines():
        yield 'data: {"jsonrpc":"2.0","id":"req-1","result":{"taskId":"task-1"}}'

    mock_resp = AsyncMock()
    mock_resp.is_success = True
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_async_client = MagicMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=mock_resp)

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)
    mock_handler.client = mock_async_client

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.proxy.common_request_processing.add_litellm_data_to_request",
                new=AsyncMock(side_effect=add_proxy_data_with_trace),
            )
        )
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        response = await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )
        if method == "tasks/resubscribe":
            async for _ in response.body_iterator:
                pass

    if method == "tasks/resubscribe":
        forwarded_headers = mock_async_client.build_request.call_args.kwargs["headers"]
    else:
        forwarded_headers = mock_handler.post.call_args.kwargs["headers"]
    assert forwarded_headers.get("X-LiteLLM-Trace-Id") == "trace-123"


@pytest.mark.asyncio
async def test_push_notification_config_set_rejects_http_url():
    """tasks/pushNotificationConfig/set must reject non-HTTPS callback URLs to prevent SSRF."""
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(
        "tasks/pushNotificationConfig/set",
        {"taskId": "task-1", "url": "http://internal-webhook.example.com/hook"},
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        with pytest.raises(HTTPException) as exc_info:
            await invoke_agent_a2a(
                agent_id="test-agent",
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

    assert exc_info.value.status_code == 400
    assert "HTTPS" in exc_info.value.detail


@pytest.mark.asyncio
async def test_push_notification_config_set_rejects_private_ip():
    """tasks/pushNotificationConfig/set must reject callback URLs pointing to private IP ranges."""
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(
        "tasks/pushNotificationConfig/set",
        {"taskId": "task-1", "url": "https://192.168.1.100/hook"},
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        with pytest.raises(HTTPException) as exc_info:
            await invoke_agent_a2a(
                agent_id="test-agent",
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

    assert exc_info.value.status_code == 400
    assert "blocked address" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_push_notification_config_set_validates_nested_url_when_top_level_present():
    """A safe top-level params.url must not let a private pushNotificationConfig.url bypass SSRF checks.

    Both URL-bearing fields are forwarded to the agent, so both must be validated independently.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(
        "tasks/pushNotificationConfig/set",
        {
            "taskId": "task-1",
            "url": "https://1.1.1.1/hook",
            "pushNotificationConfig": {"url": "https://192.168.1.100/hook"},
        },
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        with pytest.raises(HTTPException) as exc_info:
            await invoke_agent_a2a(
                agent_id="test-agent",
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

    assert exc_info.value.status_code == 400
    assert "blocked address" in exc_info.value.detail.lower()


def test_push_notification_config_set_rejects_private_dns_resolution():
    from fastapi import HTTPException

    from litellm.proxy.agent_endpoints.a2a_endpoints import (
        _validate_push_notification_url,
    )

    with patch(
        "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
        return_value=[
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("10.0.0.5", 443),
            )
        ],
    ):
        with pytest.raises(HTTPException) as exc_info:
            _validate_push_notification_url("https://webhook.example.com/hook")

    assert exc_info.value.status_code == 400
    assert "blocked address" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_push_notification_config_set_rejects_null_push_config():
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    agent = _make_agent_mock()
    mock_request = _make_request_mock(
        "tasks/pushNotificationConfig/set",
        {"taskId": "task-1", "pushNotificationConfig": None},
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u1", team_id="t1")

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        with pytest.raises(HTTPException) as exc_info:
            await invoke_agent_a2a(
                agent_id="test-agent",
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

    assert exc_info.value.status_code == 400
    assert "pushNotificationConfig must be an object" in exc_info.value.detail


@pytest.mark.asyncio
async def test_caller_identity_headers_cannot_be_spoofed_via_forwarded_headers():
    """A client must not be able to override X-LiteLLM-User-Id / X-LiteLLM-Team-Id
    by including x-a2a-<agent>-x-litellm-user-id in their request headers.
    The authenticated identity must always win."""
    from litellm.proxy._types import UserAPIKeyAuth

    upstream_response = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "task-1", "status": {"state": "completed"}},
    }
    agent = _make_agent_mock()
    mock_request = _make_request_mock("tasks/get", {"id": "task-1"})
    mock_request.headers = {
        "x-a2a-test-agent-x-litellm-user-id": "attacker-user",
        "x-a2a-test-agent-x-litellm-team-id": "attacker-team",
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test", user_id="real-user", team_id="real-team"
    )

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = upstream_response
    mock_http_response.is_success = True

    mock_handler = MagicMock()
    mock_handler.post = AsyncMock(return_value=mock_http_response)

    with ExitStack() as stack:
        for p in _base_patches(agent):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
                return_value=mock_handler,
            )
        )

        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    posted_headers = mock_handler.post.call_args.kwargs.get("headers") or {}
    assert (
        posted_headers.get("X-LiteLLM-User-Id") == "real-user"
    ), "authenticated user id must not be overridden by forwarded client headers"
    assert (
        posted_headers.get("X-LiteLLM-Team-Id") == "real-team"
    ), "authenticated team id must not be overridden by forwarded client headers"
