"""
Test appending A2A agents to model lists.

Maps to: litellm/proxy/agent_endpoints/model_list_helpers.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.proxy.agent_endpoints.model_list_helpers import (
    append_agents_to_model_group,
    append_agents_to_model_info,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.agents import AgentResponse
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)


@pytest.mark.asyncio
async def test_append_agents_to_model_group():
    """Test agents are converted to model group format with a2a/ prefix"""

    # Mock agent data
    mock_agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="my-agent",
        agent_card_params={"url": "http://example.com"},
        litellm_params=None,
    )

    # Mock AgentRequestHandler at its source location
    mock_get_allowed_agents = AsyncMock(return_value=["test-agent-id"])

    # Mock global_agent_registry
    mock_registry = Mock()
    mock_registry.get_agent_by_id = Mock(return_value=mock_agent)

    with patch(
        "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
        mock_get_allowed_agents,
    ):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            mock_registry,
        ):
            model_groups = []
            user_api_key_dict = Mock(spec=UserAPIKeyAuth)

            result = await append_agents_to_model_group(
                model_groups=model_groups,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify agent was converted with a2a/ prefix
            assert len(result) == 1
            assert result[0].model_group == "a2a/my-agent"
            assert result[0].mode == "chat"
            assert result[0].providers == ["a2a"]


@pytest.mark.asyncio
async def test_append_agents_to_model_info():
    """Test agents are converted to model info format with a2a/ prefix"""

    # Mock agent data
    mock_agent = AgentResponse(
        agent_id="agent-123",
        agent_name="test-agent",
        agent_card_params={"url": "http://example.com"},
        litellm_params=None,
        created_by="user-123",
    )

    # Mock AgentRequestHandler at its source location
    mock_get_allowed_agents = AsyncMock(return_value=["agent-123"])

    # Mock global_agent_registry
    mock_registry = Mock()
    mock_registry.get_agent_by_id = Mock(return_value=mock_agent)

    with patch(
        "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
        mock_get_allowed_agents,
    ):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            mock_registry,
        ):
            models = []
            user_api_key_dict = Mock(spec=UserAPIKeyAuth)

            result = await append_agents_to_model_info(
                models=models,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify agent was converted with a2a/ prefix
            assert len(result) == 1
            assert result[0]["model_name"] == "a2a/test-agent"
            assert result[0]["litellm_params"]["model"] == "a2a/test-agent"
            assert result[0]["litellm_params"]["custom_llm_provider"] == "a2a"
            assert result[0]["model_info"]["id"] == "agent-123"
            assert result[0]["model_info"]["mode"] == "chat"
