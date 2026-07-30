"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

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


def _router_without_a2a_models():
    """A router that knows nothing about the agent, so routing falls to the A2A branch."""
    router = Mock()
    router.model_names = ["gpt-4"]
    router.deployment_names = []
    router.has_model_id = Mock(return_value=False)
    router.model_group_alias = None
    router.router_general_settings = Mock(pass_through_all_models=False)
    router.default_deployment = None
    router.pattern_router = Mock(patterns=[])
    router.map_team_model = Mock(return_value=None)
    return router


def _registry_with_agent(**agent_kwargs):
    from litellm.types.agents import AgentResponse

    agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params=None,
        **agent_kwargs,
    )
    registry = Mock()
    registry.get_agent_by_name = Mock(return_value=agent)
    return registry


async def _route(data, registry):
    mock_acompletion = AsyncMock(return_value={"id": "test-response"})
    with patch("litellm.acompletion", mock_acompletion):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            registry,
        ):
            await route_request(
                data=data,
                llm_router=_router_without_a2a_models(),
                user_model=None,
                route_type="acompletion",
            )
    mock_acompletion.assert_called_once()
    return mock_acompletion.call_args.kwargs


@pytest.mark.asyncio
async def test_static_headers_are_sent_to_the_agent():
    """
    Regression: an agent whose upstream authenticates its callers answered fine
    over ``POST /a2a/{agent_id}`` but rejected every request routed through the
    ``a2a/<agent-name>`` chat model with a 401, because this path injected only
    ``api_base`` and dropped the agent's ``static_headers``.
    """
    registry = _registry_with_agent(static_headers={"Authorization": "Bearer upstream-secret"})

    call_kwargs = await _route(
        {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
        registry,
    )

    assert call_kwargs["headers"] == {"Authorization": "Bearer upstream-secret"}


@pytest.mark.asyncio
async def test_static_headers_win_over_caller_supplied_headers():
    """
    The admin-configured credential wins on a case-insensitive collision, and
    unrelated caller headers survive — the same precedence ``merge_agent_headers``
    applies for ``POST /a2a/{agent_id}``.
    """
    registry = _registry_with_agent(static_headers={"Authorization": "Bearer upstream-secret"})

    call_kwargs = await _route(
        {
            "model": "a2a/test-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "headers": {"authorization": "Bearer caller-supplied", "X-Trace": "keep-me"},
        },
        registry,
    )

    assert call_kwargs["headers"]["Authorization"] == "Bearer upstream-secret"
    assert call_kwargs["headers"]["X-Trace"] == "keep-me"
    assert "authorization" not in call_kwargs["headers"]


@pytest.mark.asyncio
async def test_agent_without_static_headers_sends_no_headers():
    """An agent with nothing configured must not gain a headers kwarg it never had."""
    registry = _registry_with_agent()

    call_kwargs = await _route(
        {"model": "a2a/test-agent", "messages": [{"role": "user", "content": "Hello"}]},
        registry,
    )

    assert not call_kwargs.get("headers")
