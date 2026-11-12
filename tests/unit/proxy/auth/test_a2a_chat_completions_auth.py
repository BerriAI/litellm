"""
Unit tests for A2A agent model authorization via chat completions (#50679).

Verifies that:
1. Invoking an A2A agent via model: "a2a/<agent_name>" bypasses the standard
   LLM model allowlist checks on teams and keys, and is authorized via
   AgentRequestHandler permissions.
2. Unauthorized agents return HTTP 403.
3. Standard LLM models (e.g. gpt-4o) remain strictly guarded by team/key model allowlists.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
    AgentRequestHandler,
)
from litellm.proxy.auth.auth_checks import (
    can_key_call_model,
    can_key_call_resolved_model,
    can_team_access_model,
    common_checks,
)
from litellm.types.agents import AgentResponse


@pytest.fixture
def a2a_agent_registry():
    registry = AgentRegistry()
    allowed_agent = AgentResponse(
        agent_id="test-a2a-agent-id",
        agent_name="test-a2a-agent",
        agent_card_params={"url": "http://test-a2a-agent:8080/a2a/jsonrpc"},
        litellm_params={},
    )
    denied_agent = AgentResponse(
        agent_id="denied-a2a-agent-id",
        agent_name="denied-a2a-agent",
        agent_card_params={"url": "http://denied-a2a-agent:8080/a2a/jsonrpc"},
        litellm_params={},
    )
    registry.register_agent(allowed_agent)
    registry.register_agent(denied_agent)
    return registry, allowed_agent, denied_agent


@pytest.mark.asyncio
async def test_a2a_chat_completions_authorized_with_restricted_team_models(a2a_agent_registry):
    """
    Issue #50679: A team with restricted models allowlist (e.g. ['gpt-4o-mini'])
    must be able to invoke a granted A2A agent via model: 'a2a/test-a2a-agent'
    without triggering team_model_access_denied.
    """
    registry, allowed_agent, _ = a2a_agent_registry

    # Team has only 'gpt-4o-mini' in models, but grants 'test-a2a-agent-id' in object_permission
    team = LiteLLM_TeamTable(
        team_id="test-team-1",
        models=["gpt-4o-mini"],
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-1", agents=[allowed_agent.agent_id]),
    )

    valid_token = UserAPIKeyAuth(
        token="sk-test-key",
        team_id="test-team-1",
        models=[],
    )

    mock_request = Mock(spec=Request)
    mock_request.headers = {}
    mock_request.query_params = {}
    mock_request.method = "POST"
    mock_request.state = Mock()

    request_body = {
        "model": "a2a/test-a2a-agent",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
        registry,
    ):
        # common_checks should pass without raising team_model_access_denied
        result = await common_checks(
            request_body=request_body,
            team_object=team,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=0.0,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=None,
            proxy_logging_obj=Mock(),
            valid_token=valid_token,
            request=mock_request,
            skip_budget_checks=True,
        )
        assert result is True


@pytest.mark.asyncio
async def test_a2a_chat_completions_denied_for_unauthorized_agent(a2a_agent_registry):
    """
    A key/team that has NOT been granted access to an agent should receive 403
    when attempting to call model: 'a2a/denied-a2a-agent'.
    """
    registry, allowed_agent, denied_agent = a2a_agent_registry

    team = LiteLLM_TeamTable(
        team_id="test-team-2",
        models=["gpt-4o-mini"],
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-2", agents=[allowed_agent.agent_id]),
    )

    valid_token = UserAPIKeyAuth(
        token="sk-test-key-2",
        team_id="test-team-2",
        models=[],
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-2-key", agents=[allowed_agent.agent_id]),
    )

    mock_request = Mock(spec=Request)
    mock_request.headers = {}
    mock_request.query_params = {}
    mock_request.method = "POST"
    mock_request.state = Mock()

    request_body = {
        "model": "a2a/denied-a2a-agent",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
        registry,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await common_checks(
                request_body=request_body,
                team_object=team,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=0.0,
                general_settings={},
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=Mock(),
                valid_token=valid_token,
                request=mock_request,
                skip_budget_checks=True,
            )
        assert exc_info.value.status_code == 403
        assert "is not allowed for your key/team" in exc_info.value.detail


@pytest.mark.asyncio
async def test_standard_llm_models_still_enforce_team_model_allowlist():
    """
    Ensure regular LLM models (e.g. 'gpt-4o') are still strictly checked against
    the team's model allowlist and raise team_model_access_denied if not in the list.
    """
    team = LiteLLM_TeamTable(
        team_id="test-team-3",
        models=["gpt-4o-mini"],
    )

    valid_token = UserAPIKeyAuth(
        token="sk-test-key-3",
        team_id="test-team-3",
    )

    mock_request = Mock(spec=Request)
    mock_request.headers = {}
    mock_request.query_params = {}
    mock_request.method = "POST"
    mock_request.state = Mock()

    request_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with pytest.raises(ProxyException) as exc_info:
        await common_checks(
            request_body=request_body,
            team_object=team,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=0.0,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=None,
            proxy_logging_obj=Mock(),
            valid_token=valid_token,
            request=mock_request,
            skip_budget_checks=True,
        )
    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied


@pytest.mark.asyncio
async def test_can_key_call_model_allows_granted_a2a_agent(a2a_agent_registry):
    """
    can_key_call_model should pass for an A2A model when the key has agent grants,
    even if key.models only contains LLM models like ['gpt-4o-mini'].
    """
    registry, allowed_agent, denied_agent = a2a_agent_registry

    valid_token = UserAPIKeyAuth(
        token="sk-key-with-restricted-models",
        models=["gpt-4o-mini"],
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-key", agents=[allowed_agent.agent_id]),
    )

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
        registry,
    ):
        # Should succeed for allowed agent
        assert (
            await can_key_call_model(
                model="a2a/test-a2a-agent",
                llm_model_list=None,
                valid_token=valid_token,
                llm_router=None,
            )
            is True
        )

        # Should raise 403 for denied agent
        with pytest.raises(HTTPException) as exc_info:
            await can_key_call_model(
                model="a2a/denied-a2a-agent",
                llm_model_list=None,
                valid_token=valid_token,
                llm_router=None,
            )
        assert exc_info.value.status_code == 403
        assert "is not allowed for your key/team" in exc_info.value.detail


@pytest.mark.asyncio
async def test_can_team_access_model_passes_for_a2a_model():
    """
    can_team_access_model returns True for an a2a/ model so it is not blocked by
    team.models allowlist.
    """
    team = LiteLLM_TeamTable(
        team_id="team-xyz",
        models=["gpt-4o-mini"],
    )
    assert await can_team_access_model(model="a2a/any-agent", team_object=team, llm_router=None) is True
