"""
Test A2A cost calculator with cost_per_query parameter.
"""

import asyncio
from typing import Any, AsyncIterator, Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


async def _reset_callbacks_and_settle_pending_logs() -> None:
    litellm.logging_callback_manager._reset_all_callbacks()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)


def _make_send_message_request(request_id: str, user_text: str = "Hello"):
    from a2a.compat.v0_3.types import MessageSendParams, SendMessageRequest

    return SendMessageRequest(
        id=request_id,
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": user_text}],
                "messageId": "msg-1",
            }
        ),
    )


async def _mock_execute_a2a_send(
    a2a_client: Any,
    request: Any,
    **kwargs: Any,
) -> Any:
    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(
        return_value={
            "id": request.id,
            "jsonrpc": "2.0",
            "result": {"status": "completed"},
        }
    )
    return mock_response


async def _mock_execute_a2a_send_with_assistant_reply(
    a2a_client: Any,
    request: Any,
    **kwargs: Any,
) -> Any:
    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(
        return_value={
            "id": request.id,
            "jsonrpc": "2.0",
            "result": {
                "status": {"state": "completed"},
                "message": {
                    "role": "assistant",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Hello! I am your assistant. How can I help you today?",
                        }
                    ],
                    "messageId": "msg-456",
                },
            },
        }
    )
    return mock_response


def _make_streaming_request(request_id: str):
    from a2a.compat.v0_3.types import MessageSendParams, SendStreamingMessageRequest

    return SendStreamingMessageRequest(
        id=request_id,
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
                "messageId": "msg-1",
            }
        ),
    )


async def _mock_stream_messages(a2a_client: Any, request: Any) -> AsyncIterator[Any]:
    from a2a.compat.v0_3.types import (
        Message,
        Part,
        Role,
        SendStreamingMessageResponse,
        SendStreamingMessageSuccessResponse,
        TextPart,
    )

    msg = Message(
        message_id="msg-agent",
        role=Role.agent,
        parts=[Part(root=TextPart(kind="text", text="hello"))],
        kind="message",
    )
    for _ in range(2):
        yield SendStreamingMessageResponse(root=SendStreamingMessageSuccessResponse(id=request.id, result=msg))


class CostLogger(CustomLogger):
    """Custom logger to capture response_cost."""

    def __init__(self):
        self.response_cost: Optional[float] = None
        self.logged: asyncio.Event = asyncio.Event()
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        slp = kwargs.get("standard_logging_object")
        if slp:
            self.response_cost = (
                slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)
            )
        self.logged.set()


@pytest.mark.asyncio
async def test_asend_message_uses_cost_per_query(monkeypatch):
    """
    Test that asend_message uses cost_per_query param for response_cost.
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    await _reset_callbacks_and_settle_pending_logs()
    cost_logger = CostLogger()
    monkeypatch.setattr(litellm, "callbacks", [cost_logger])

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_send_message_request("test-123")

    # Call asend_message with cost_per_query
    with patch(
        "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
        new=_mock_execute_a2a_send,
    ):
        await asend_message(
            a2a_client=mock_client,
            request=mock_request,
            cost_per_query=0.05,
        )

    await asyncio.sleep(0.1)

    assert cost_logger.response_cost == 0.05


@pytest.mark.asyncio
async def test_asend_message_uses_cost_per_query_from_litellm_params_dict(monkeypatch):
    """
    Proxy passes agent pricing as the litellm_params dict param (not top-level
    kwargs). Regression for cost_per_query landing at $0 on the native path.
    """
    from litellm.a2a_protocol import asend_message

    await _reset_callbacks_and_settle_pending_logs()
    cost_logger = CostLogger()
    monkeypatch.setattr(litellm, "callbacks", [cost_logger])

    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_send_message_request("test-123")

    with patch(
        "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
        new=_mock_execute_a2a_send,
    ):
        await asend_message(
            a2a_client=mock_client,
            request=mock_request,
            litellm_params={
                "cost_per_query": 0.5,
                "input_cost_per_token": 0.099999,
                "output_cost_per_token": 0.1,
            },
        )

    await asyncio.sleep(0.1)

    assert cost_logger.response_cost == 0.5


class TokenAndCostLogger(CustomLogger):
    """Custom logger to capture both token counts and cost."""

    def __init__(self):
        self.response_cost: Optional[float] = None
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        slp = kwargs.get("standard_logging_object")
        if slp:
            self.response_cost = (
                slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)
            )
            self.prompt_tokens = (
                slp.get("prompt_tokens") if isinstance(slp, dict) else getattr(slp, "prompt_tokens", None)
            )
            self.completion_tokens = (
                slp.get("completion_tokens") if isinstance(slp, dict) else getattr(slp, "completion_tokens", None)
            )


@pytest.mark.asyncio
async def test_asend_message_uses_input_output_cost_per_token(monkeypatch):
    """
    Test that asend_message calculates cost using input_cost_per_token and output_cost_per_token.
    Validates exact cost calculation: cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    await _reset_callbacks_and_settle_pending_logs()
    token_cost_logger = TokenAndCostLogger()
    monkeypatch.setattr(litellm, "callbacks", [token_cost_logger])

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_send_message_request("test-123", user_text="Hello, what can you do?")

    # Define specific cost per token values
    input_cost_per_token = 0.00001  # $0.01 per 1000 tokens
    output_cost_per_token = 0.00002  # $0.02 per 1000 tokens

    with patch(
        "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
        new=_mock_execute_a2a_send_with_assistant_reply,
    ):
        await asend_message(
            a2a_client=mock_client,
            request=mock_request,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
        )

    await asyncio.sleep(0.1)

    # Get actual token counts from logger
    prompt_tokens = token_cost_logger.prompt_tokens
    completion_tokens = token_cost_logger.completion_tokens
    response_cost = token_cost_logger.response_cost

    print(f"\n=== Token-Based Cost Results ===")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    print(f"input_cost_per_token: {input_cost_per_token}")
    print(f"output_cost_per_token: {output_cost_per_token}")
    print(f"response_cost: {response_cost}")

    # Verify tokens were captured
    assert prompt_tokens is not None, "prompt_tokens should be captured"
    assert completion_tokens is not None, "completion_tokens should be captured"
    assert response_cost is not None, "response_cost should be captured"

    # Calculate expected cost
    expected_cost = (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)
    print(f"expected_cost: {expected_cost}")

    # Verify exact cost calculation
    assert response_cost == expected_cost, f"response_cost {response_cost} should equal expected {expected_cost}"


class AgentIdLogger(CustomLogger):
    """Custom logger to capture agent_id from kwargs."""

    def __init__(self):
        self.agent_id: Optional[str] = None
        self.kwargs: Optional[dict] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.kwargs = kwargs
        self.agent_id = kwargs.get("agent_id")


@pytest.mark.asyncio
async def test_asend_message_passes_agent_id_to_callback(monkeypatch):
    """
    Test that asend_message passes agent_id to callbacks via kwargs.
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    await _reset_callbacks_and_settle_pending_logs()
    agent_id_logger = AgentIdLogger()
    monkeypatch.setattr(litellm, "callbacks", [agent_id_logger])

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_send_message_request("test-123")

    test_agent_id = "agent-uuid-12345"

    # Call asend_message with agent_id
    with patch(
        "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
        new=_mock_execute_a2a_send,
    ):
        await asend_message(
            a2a_client=mock_client,
            request=mock_request,
            agent_id=test_agent_id,
        )

    await asyncio.sleep(0.1)

    # Verify agent_id was passed to callback
    assert agent_id_logger.agent_id == test_agent_id, (
        f"Expected agent_id '{test_agent_id}', got '{agent_id_logger.agent_id}'"
    )


class MetadataLogger(CustomLogger):
    """Custom logger to capture metadata from kwargs for proxy spend tracking."""

    def __init__(self):
        self.metadata: Optional[dict] = None
        self.litellm_params: Optional[dict] = None
        self.user_api_key: Optional[str] = None
        self.user_id: Optional[str] = None
        self.team_id: Optional[str] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.litellm_params = kwargs.get("litellm_params", {})
        self.metadata = self.litellm_params.get("metadata", {})
        self.user_api_key = self.metadata.get("user_api_key")
        self.user_id = self.metadata.get("user_api_key_user_id")
        self.team_id = self.metadata.get("user_api_key_team_id")


@pytest.mark.asyncio
async def test_asend_message_streaming_propagates_metadata():
    """
    Test that asend_message_streaming propagates metadata to logging object.
    This ensures user_api_key, user_id, team_id are available for SpendLogs.
    """
    from litellm.a2a_protocol import asend_message_streaming

    # Setup logger
    await _reset_callbacks_and_settle_pending_logs()
    metadata_logger = MetadataLogger()
    litellm.logging_callback_manager.add_litellm_async_success_callback(metadata_logger)

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_streaming_request("test-stream-metadata")

    # Metadata from proxy (contains user_api_key, user_id, team_id for SpendLogs)
    test_metadata = {
        "user_api_key": "sk-test-key-hash-12345",
        "user_api_key_user_id": "user-uuid-123",
        "user_api_key_team_id": "team-uuid-456",
    }

    # Consume streaming response with metadata
    chunks = []
    with patch(
        "litellm.a2a_protocol.main._stream_messages",
        new=_mock_stream_messages,
    ):
        async for chunk in asend_message_streaming(
            a2a_client=mock_client,
            request=mock_request,
            metadata=test_metadata,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.2)

    # Verify metadata was propagated to callback
    assert metadata_logger.user_api_key == "sk-test-key-hash-12345"
    assert metadata_logger.user_id == "user-uuid-123"
    assert metadata_logger.team_id == "team-uuid-456"


@pytest.mark.asyncio
async def test_asend_message_streaming_triggers_callbacks():
    """
    Test that asend_message_streaming triggers callbacks after stream completes.
    """
    from litellm.a2a_protocol import asend_message_streaming

    # Setup logger - must use logging_callback_manager to properly register
    await _reset_callbacks_and_settle_pending_logs()
    callback_logger = AgentIdLogger()
    litellm.logging_callback_manager.add_litellm_async_success_callback(callback_logger)
    litellm.logging_callback_manager.add_litellm_success_callback(callback_logger)

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    mock_request = _make_streaming_request("test-stream-123")

    test_agent_id = "test-agent-id-streaming"

    # Consume streaming response
    chunks = []
    with patch(
        "litellm.a2a_protocol.main._stream_messages",
        new=_mock_stream_messages,
    ):
        async for chunk in asend_message_streaming(
            a2a_client=mock_client,
            request=mock_request,
            agent_id=test_agent_id,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.2)

    # Verify chunks were received
    assert len(chunks) == 2

    # Verify callbacks WERE triggered after stream completed
    assert callback_logger.kwargs is not None, "Streaming should trigger callbacks after completion"
    assert callback_logger.agent_id == test_agent_id, (
        f"Expected agent_id '{test_agent_id}', got '{callback_logger.agent_id}'"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("claimed_fee", (None, -1000.0, 0.0, 99.0))
@pytest.mark.parametrize("fee_field", ("cost_per_query", "litellm_params"))
@pytest.mark.parametrize("changed_after_admission", (False, True))
@pytest.mark.parametrize("configured_fee", (None, 0.0, 0.25))
async def test_chat_adapter_settles_the_admitted_agent_fee_without_model_pricing(
    monkeypatch: pytest.MonkeyPatch, stream: bool, claimed_fee: float | None,
    changed_after_admission: bool, configured_fee: float | None, fee_field: str
) -> None:
    import json
    from typing import Final

    import httpx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation
    from litellm.types.agents import AgentResponse

    await _reset_callbacks_and_settle_pending_logs()
    logger: Final = CostLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    target: Final = AgentResponse(
        agent_id="fee-target", agent_name="fee-target",
        agent_card_params={"url": "https://agent.test/", "capabilities": {"streaming": True}},
        litellm_params={"cost_per_query": configured_fee} if configured_fee is not None else {},
    )
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(target)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    auth: Final = UserAPIKeyAuth(user_role="proxy_admin")
    await prepare_agent_invocation(auth, "fee-target", None)
    assert auth.agent_invocation_cost == configured_fee
    if changed_after_admission:
        registry.deregister_agent(target.agent_name)
        registry.register_agent(target.model_copy(update={"litellm_params": {"cost_per_query": 0.5}}))

    def reply(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body["method"] == ("message/stream" if stream else "message/send")
        result: Final = {
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"kind": "message", "messageId": "reply", "role": "agent",
                       "parts": [{"kind": "text", "text": "Paid reply"}]},
        }
        if stream:
            return httpx.Response(200, text=f"data: {json.dumps(result)}\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=result)

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(reply))
    try:
        pending: Final = await route_a2a_agent_request(
            data={"model": "a2a/fee-target", "messages": [{"role": "user", "content": "Hello"}],
                  "stream": stream, "client": client,
                  **({fee_field: claimed_fee if fee_field == "cost_per_query" else {"cost_per_query": claimed_fee}}
                     if claimed_fee is not None else {})},
            route_type="acompletion", user_api_key_dict=auth,
        )
        response: Final = await pending
        if stream:
            chunks: Final = tuple([chunk async for chunk in response])
            assert any(chunk.choices[0].delta.content == "Paid reply" for chunk in chunks)
        else:
            assert response.choices[0].message.content == "Paid reply"
        await asyncio.wait_for(logger.logged.wait(), timeout=10.0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
        if configured_fee is None:
            assert logger.response_cost in (None, 0.0)
        else:
            assert logger.response_cost == pytest.approx(configured_fee)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted_target", (None, "other"))
async def test_chat_agent_dispatch_rejects_missing_or_different_admission(
    monkeypatch: pytest.MonkeyPatch,
    admitted_target: str | None,
) -> None:
    from typing import Final
    from unittest.mock import Mock

    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation
    from litellm.types.agents import AgentResponse

    registry: Final = agent_registry.AgentRegistry()
    for name in ("paid", "other"):
        registry.register_agent(
            AgentResponse(
                agent_id=name,
                agent_name=name,
                agent_card_params={"url": "https://agent.test/"},
                litellm_params={"cost_per_query": 0.25},
            )
        )
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    auth: Final = UserAPIKeyAuth(user_role="proxy_admin")
    if admitted_target is not None:
        await prepare_agent_invocation(auth, admitted_target, None)
    provider: Final = Mock(return_value=None)
    monkeypatch.setattr(litellm, "acompletion", provider)
    with pytest.raises(HTTPException) as exc:
        await route_a2a_agent_request(
            data={"model": "a2a/paid", "messages": [{"role": "user", "content": "Hello"}]},
            route_type="acompletion",
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 503
    assert "admission" in str(exc.value.detail).lower()
    provider.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("configured_fee", (None, 0.0, 0.25))
async def test_chat_adapter_preserves_model_pricing_without_a_fixed_fee(
    monkeypatch: pytest.MonkeyPatch, stream: bool, configured_fee: float | None
) -> None:
    import json
    from typing import Final

    import httpx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation
    from litellm.types.agents import AgentResponse

    await _reset_callbacks_and_settle_pending_logs()
    logger: Final = CostLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(litellm, "model_cost", {
        **litellm.model_cost,
        "a2a/token-target": {
            "input_cost_per_token": 0.0, "output_cost_per_token": 0.125,
            "litellm_provider": "a2a", "mode": "chat",
        },
    })
    target: Final = AgentResponse(
        agent_id="token-target", agent_name="token-target",
        agent_card_params={"url": "https://agent.test/", "capabilities": {"streaming": True}},
        litellm_params={"cost_per_query": configured_fee} if configured_fee is not None else {},
    )
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(target)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    auth: Final = UserAPIKeyAuth(user_role="proxy_admin")
    auth.billing_agent_policy = AgentResponse(agent_id="caller", agent_name="caller", agent_card_params={})
    await prepare_agent_invocation(auth, target.agent_id, None)

    def reply(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        result: Final = {
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"kind": "message", "messageId": "reply", "role": "agent",
                       "parts": [{"kind": "text", "text": "Hello"}]},
        }
        if stream:
            return httpx.Response(200, text=f"data: {json.dumps(result)}\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=result)

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(reply))
    try:
        pending: Final = await route_a2a_agent_request(
            data={"model": "a2a/token-target", "messages": [{"role": "user", "content": "Hi"}],
                  "stream": stream, "client": client},
            route_type="acompletion", user_api_key_dict=auth,
        )
        response: Final = await pending
        if stream:
            chunks: Final = tuple([chunk async for chunk in response])
            assert any(chunk.choices[0].delta.content == "Hello" for chunk in chunks)
        else:
            assert response.choices[0].message.content == "Hello"
        await asyncio.wait_for(logger.logged.wait(), timeout=10.0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
        assert logger.response_cost == pytest.approx(0.125 if configured_fee is None else configured_fee)
    finally:
        await client.close()
