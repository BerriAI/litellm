"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.a2a_routing import (
    _route_registered_provider,
    merge_a2a_agent_guardrails_before_hooks,
    route_a2a_agent_request,
)
from litellm.proxy.route_llm_request import route_request


@pytest.mark.asyncio
async def test_route_a2a_model_bypasses_router():
    """Test that a2a/ prefixed models bypass router and go directly to litellm with api_base"""

    # Mock data for chat completion with a2a model
    data = {
        "model": "a2a/test-agent",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock router that doesn't have the a2a model
    mock_router = Mock()
    mock_router.model_names = ["gpt-4", "gpt-3.5-turbo"]
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)

    # Mock agent in registry
    from litellm.types.agents import AgentResponse

    mock_agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params=None,
    )

    mock_registry = Mock()
    mock_registry.get_agent_by_id = Mock(return_value=None)
    mock_registry.get_agent_by_name = Mock(return_value=mock_agent)

    # Mock litellm.acompletion to verify it's called
    mock_acompletion = AsyncMock(return_value={"id": "test-response"})

    with patch("litellm.acompletion", mock_acompletion):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            mock_registry,
        ):
            await route_request(
                data=data,
                llm_router=mock_router,
                user_model=None,
                route_type="acompletion",
            )

            # Verify litellm.acompletion was called with api_base injected
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "a2a/test-agent"
            assert call_kwargs["api_base"] == "http://agent.example.com"


@pytest.mark.asyncio
async def test_route_a2a_model_uses_registered_provider():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={
            "custom_llm_provider": "pydantic_ai_agents",
            "guardrails": ["agent-guardrail"],
        },
        static_headers={"Authorization": "Bearer static"},
        extra_headers=["X-Tenant"],
    )
    data = {
        "model": "a2a/test-agent",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": ["request-guardrail"],
        "max_tokens": 32,
        "temperature": 0.2,
        "timeout": 12.0,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "output_config": {"format": "json"},
        "prompt_cache_key": "cache-key",
        "safety_identifier": "safety-id",
        "proxy_server_request": {
            "headers": {
                "x-tenant": "tenant-1",
                "x-a2a-test-agent-x-run": "run-1",
            }
        },
    }
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": "Hello back"}],
            "messageId": "message-id",
        },
    }

    with (
        patch(  # test-quality-ok: registry lookup is the routing seam
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(  # test-quality-ok: access control is outside this routing test
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(  # test-quality-ok: provider dispatch is the tested seam
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
        patch(  # test-quality-ok: generic dispatch must stay unused
            "litellm.acompletion", AsyncMock()
        ) as generic_completion,
    ):
        call = await route_a2a_agent_request(data, "acompletion")
        response = await call

    bridge.assert_awaited_once()
    generic_completion.assert_not_called()
    assert response.choices[0].message.content == "Hello back"
    bridge_kwargs = bridge.await_args.kwargs
    assert bridge_kwargs["litellm_params"]["max_tokens"] == 32
    assert bridge_kwargs["litellm_params"]["temperature"] == 0.2
    assert bridge_kwargs["litellm_params"]["timeout"] == 12.0
    assert bridge_kwargs["litellm_params"]["tools"] == data["tools"]
    assert bridge_kwargs["litellm_params"]["output_config"] == data["output_config"]
    assert bridge_kwargs["litellm_params"]["prompt_cache_key"] == data["prompt_cache_key"]
    assert bridge_kwargs["litellm_params"]["safety_identifier"] == data["safety_identifier"]
    assert bridge_kwargs["litellm_params"]["guardrails"] == ["request-guardrail", "agent-guardrail"]
    assert bridge_kwargs["litellm_params"]["extra_headers"] == {
        "X-Tenant": "tenant-1",
        "x-run": "run-1",
        "Authorization": "Bearer static",
    }


@pytest.mark.asyncio
async def test_route_a2a_cardless_bedrock_agentcore_uses_registered_model():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={},
        litellm_params={
            "custom_llm_provider": "bedrock",
            "model": "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
        },
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        await call

    assert bridge.await_args.kwargs["api_base"] is None


@pytest.mark.asyncio
async def test_route_a2a_registered_provider_uses_configured_api_base_without_card_url():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={},
        litellm_params={
            "custom_llm_provider": "langflow",
            "model": "flow",
            "api_base": "https://flow.example.com",
        },
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        await call

    assert bridge.await_args.kwargs["api_base"] == "https://flow.example.com"


@pytest.mark.asyncio
async def test_registered_provider_response_preserves_multiple_choices():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "choices": [
            {"index": 0, "message": {"parts": [{"kind": "text", "text": "first"}]}},
            {"index": 1, "message": {"parts": [{"kind": "text", "text": "second"}]}},
        ],
        "result": {},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ),
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        response = await call

    assert [choice.message.content for choice in response.choices] == ["first", "second"]


@pytest.mark.asyncio
async def test_route_a2a_cardless_watsonx_orchestrate_uses_registered_model():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={},
        litellm_params={
            "custom_llm_provider": "watsonx_orchestrate",
            "model": "agent",
            "cp4d_host": "https://wxo.example.com",
            "instance_id": "instance",
        },
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        await call

    assert bridge.await_args.kwargs["api_base"] is None


@pytest.mark.asyncio
async def test_route_a2a_registered_provider_preserves_identity_headers():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            {
                "model": "a2a/test-agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "proxy_server_request": {
                    "headers": {
                        "x-a2a-test-agent-x-litellm-user-id": "attacker",
                        "x-a2a-test-agent-x-litellm-team-id": "attacker-team",
                    }
                },
            },
            "acompletion",
            user_api_key_dict=UserAPIKeyAuth(user_id="trusted-user", team_id="trusted-team"),
        )
        await call

    headers = bridge.await_args.kwargs["agent_extra_headers"]
    assert headers["X-LiteLLM-User-Id"] == "trusted-user"
    assert headers["X-LiteLLM-Team-Id"] == "trusted-team"
    assert "x-litellm-user-id" not in {key.lower() for key in headers if key != "X-LiteLLM-User-Id"}


@pytest.mark.asyncio
async def test_route_a2a_registered_provider_preserves_messages_and_session():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import A2A_USER_API_KEY_HASH_PARAM
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "langflow", "model": "flow"},
    )
    data = {
        "model": "a2a/test-agent",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ],
        "litellm_session_id": "session-1",
    }
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            data,
            "acompletion",
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        )
        await call

    bridge_kwargs = bridge.await_args.kwargs
    assert bridge_kwargs["params"]["messages"] == data["messages"]
    assert bridge_kwargs["params"]["message"]["contextId"] == "session-1"
    assert bridge_kwargs["litellm_params"][A2A_USER_API_KEY_HASH_PARAM] == "hashed-key"


@pytest.mark.asyncio
async def test_route_a2a_requires_inbound_trace_id():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={
            "custom_llm_provider": "pydantic_ai_agents",
            "require_trace_id_on_calls_to_agent": True,
        },
    )

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
    ):
        with pytest.raises(HTTPException, match="requires x-litellm-trace-id"):
            await route_a2a_agent_request(
                {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
                "acompletion",
            )


@pytest.mark.asyncio
async def test_route_a2a_resolves_databricks_oauth_headers():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "databricks", "databricks_oauth": {"client_id": "id"}},
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {"kind": "message", "parts": [{"kind": "text", "text": "Hello back"}]},
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.proxy.agent_endpoints.databricks_oauth.resolve_databricks_app_auth_header",
            AsyncMock(return_value={"Authorization": "Bearer minted"}),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ) as bridge,
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        await call

    assert bridge.await_args.kwargs["agent_extra_headers"]["Authorization"] == "Bearer minted"


@pytest.mark.asyncio
async def test_registered_provider_response_preserves_tool_calls():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
    )
    bridge_response = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {
            "kind": "message",
            "parts": [],
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
            "finish_reason": "tool_calls",
        },
    }

    with (
        patch(
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=bridge_response),
        ),
    ):
        call = await route_a2a_agent_request(
            {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
            "acompletion",
        )
        response = await call

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].id == "call-1"


@pytest.mark.asyncio
async def test_a2a_agent_guardrails_merge_before_hooks():
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"guardrails": ["agent-guardrail"]},
    )
    with patch(
        "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
        AsyncMock(return_value=agent),
    ):
        merged = await merge_a2a_agent_guardrails_before_hooks(
            {"model": "a2a/test-agent", "guardrails": ["request-guardrail"]}
        )

    assert merged["guardrails"] == ["request-guardrail", "agent-guardrail"]


@pytest.mark.asyncio
async def test_route_a2a_stream_uses_registered_provider():
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
    )
    logging_obj = Mock(spec=Logging)
    logging_obj.model_call_details = {}
    data = {
        "model": "a2a/test-agent",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "litellm_logging_obj": logging_obj,
    }
    provider_stream = object()
    completion_stream = object()
    wrapper = object()

    with (
        patch(  # test-quality-ok: registry lookup is the routing seam
            "litellm.proxy.common_utils.registry_read_through.get_agent_with_read_through",
            AsyncMock(return_value=agent),
        ),
        patch(  # test-quality-ok: access control is outside this routing test
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            AsyncMock(return_value=True),
        ),
        patch(  # test-quality-ok: provider dispatch is the tested seam
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_streaming",
            Mock(return_value=provider_stream),
        ) as bridge,
        patch(  # test-quality-ok: iterator wiring is the tested seam
            "litellm.llms.a2a.chat.streaming_iterator.A2AModelResponseIterator",
            Mock(return_value=completion_stream),
        ),
        patch(  # test-quality-ok: wrapper wiring is the tested seam
            "litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper",
            Mock(return_value=wrapper),
        ) as stream_wrapper,
        patch(  # test-quality-ok: generic dispatch must stay unused
            "litellm.acompletion", AsyncMock()
        ) as generic_completion,
    ):
        call = await route_a2a_agent_request(data, "acompletion")
        response = await call

    bridge.assert_called_once()
    stream_wrapper.assert_called_once_with(
        completion_stream=completion_stream,
        model="a2a/test-agent",
        custom_llm_provider="a2a",
        logging_obj=logging_obj,
        stream_options=None,
    )
    generic_completion.assert_not_called()
    assert response is wrapper


@pytest.mark.asyncio
async def test_registered_provider_logging_uses_provider_model_for_builtin_pricing():
    class FakeLogging:
        def __init__(self) -> None:
            self.model_call_details = {"litellm_params": {}}
            self.litellm_params = self.model_call_details["litellm_params"]
            self.custom_pricing = False

    logging_obj = FakeLogging()
    response = {"result": {"message": {"parts": [{"kind": "text", "text": "hello"}]}}}
    with (
        patch("litellm.litellm_core_utils.litellm_logging.Logging", FakeLogging),
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2ACompletionBridgeHandler.handle_non_streaming",
            AsyncMock(return_value=response),
        ),
    ):
        await _route_registered_provider(
            data={
                "messages": [{"role": "user", "content": "hello"}],
                "litellm_logging_obj": logging_obj,
            },
            model_name="a2a/agent",
            api_base="https://provider.example",
            litellm_params={"model": "gpt-4o", "custom_llm_provider": "openai"},
            static_headers=None,
        )

    assert logging_obj.model_call_details["model"] == "gpt-4o"
    assert logging_obj.model_call_details["custom_llm_provider"] == "openai"
    assert logging_obj.model_call_details["litellm_params"]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_route_non_a2a_model_raises_error_if_not_in_router():
    """Test that non-a2a models that aren't in router raise an error"""

    # Mock data for chat completion with model not in router
    data = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock router without the model
    mock_router = Mock()
    mock_router.model_names = ["gpt-4", "gpt-3.5-turbo"]
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)

    # Should raise ProxyModelNotFoundError
    from litellm.proxy.route_llm_request import ProxyModelNotFoundError

    with pytest.raises(ProxyModelNotFoundError):
        await route_request(
            data=data,
            llm_router=mock_router,
            user_model=None,
            route_type="acompletion",
        )


class _DbAgentRow:
    def __init__(self, agent_id: str, agent_name: str) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.object_permission = None
        self.spend = 0.0

    def model_dump(self):
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_card_params": {"name": self.agent_name, "url": "http://sibling-db-agent.example.com"},
            "litellm_params": {},
            "object_permission": None,
            "spend": self.spend,
        }


def _router_without_models():
    mock_router = Mock()
    mock_router.model_names = []
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.team_public_model_names = []
    return mock_router


@pytest.mark.asyncio
async def test_route_a2a_model_read_through_recovers_agent_created_on_sibling_replica(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agent_name = "a2a-sibling-replica-agent"
    prisma_client = Mock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        side_effect=[None, _DbAgentRow("a2a-sibling-replica-agent-id", agent_name)]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    original_agents = list(global_agent_registry.agent_list)
    original_config_agents = getattr(global_agent_registry, "config_agents", ())
    global_agent_registry.agent_list = []
    global_agent_registry.config_agents = ()

    data = {
        "model": f"a2a/{agent_name}",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    mock_acompletion = AsyncMock(return_value={"id": "read-through-response"})

    try:
        with patch("litellm.acompletion", mock_acompletion):
            await route_request(
                data=data,
                llm_router=_router_without_models(),
                user_model=None,
                route_type="acompletion",
            )
    finally:
        global_agent_registry.agent_list = original_agents
        global_agent_registry.config_agents = original_config_agents

    mock_acompletion.assert_called_once()
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == f"a2a/{agent_name}"
    assert call_kwargs["api_base"] == "http://sibling-db-agent.example.com"
    prisma_client.db.litellm_agentstable.find_unique.assert_awaited()
