"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
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
            result = await route_request(
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
    import litellm.proxy.proxy_server as proxy_server
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
