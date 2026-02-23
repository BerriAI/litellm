from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints import endpoints as agent_endpoints
from litellm.proxy.agent_endpoints.endpoints import (
    get_agent_daily_activity,
    router,
    user_api_key_auth,
)
from litellm.types.agents import AgentResponse


def _sample_agent_card_params() -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "Test Agent",
        "description": "desc",
        "url": "http://localhost",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _sample_agent_config() -> dict:
    return {
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {"make_public": False},
    }


def _sample_agent_response(
    agent_id: str = "agent-123", agent_name: str = "Test Agent"
) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_card_params=_sample_agent_card_params(),
        litellm_params={"make_public": False},
    )


app = FastAPI()
app.include_router(router)
app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
    user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
)
client = TestClient(app)


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:
        yield mock


@pytest.fixture
def mock_user_api_key_auth():
    with patch("litellm.proxy.agent_endpoints.endpoints.user_api_key_auth") as mock:
        mock.return_value = UserAPIKeyAuth(
            user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
        )
        yield mock


def test_update_agent_success(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    existing_agent = {
        "agent_id": "agent-123",
        "agent_name": "Existing Agent",
        "agent_card_params": _sample_agent_card_params(),
    }
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=existing_agent
    )

    mock_registry = MagicMock()
    mock_registry.update_agent_in_db = AsyncMock(
        return_value=_sample_agent_response(agent_id="agent-123")
    )
    mock_registry.deregister_agent = MagicMock()
    mock_registry.register_agent = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.put(
        "/v1/agents/agent-123",
        json=_sample_agent_config(),
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["agent_id"] == "agent-123"
    assert response.json()["agent_name"] == "Test Agent"


def test_update_agent_not_found(
    mock_prisma_client, mock_user_api_key_auth, monkeypatch
):
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    mock_registry = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.put(
        "/v1/agents/missing-agent",
        json=_sample_agent_config(),
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found" in response.json()["detail"]


def test_get_agent_by_id_not_found(
    mock_prisma_client, mock_user_api_key_auth, monkeypatch
):
    mock_registry = MagicMock()
    mock_registry.get_agent_by_id = MagicMock(return_value=None)
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    response = client.get(
        "/v1/agents/missing-agent", headers={"Authorization": "Bearer test-key"}
    )

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found" in response.json()["detail"]


def test_delete_agent_not_found(
    mock_prisma_client, mock_user_api_key_auth, monkeypatch
):
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    mock_registry = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.delete(
        "/v1/agents/missing-agent", headers={"Authorization": "Bearer test-key"}
    )

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found in DB." in response.json()["detail"]


def test_agent_error_schema_consistency(
    mock_prisma_client, mock_user_api_key_auth, monkeypatch
):
    mock_registry = MagicMock()
    mock_registry.get_agent_by_id = MagicMock(return_value=None)
    mock_registry.update_agent_in_db = AsyncMock(side_effect=Exception("should not run"))
    mock_registry.delete_agent_from_db = AsyncMock(side_effect=Exception("should not run"))
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    missing_agent_id = "missing-agent"
    responses = [
        client.get(
            f"/v1/agents/{missing_agent_id}",
            headers={"Authorization": "Bearer test-key"},
        ),
        client.put(
            f"/v1/agents/{missing_agent_id}",
            json=_sample_agent_config(),
            headers={"Authorization": "Bearer test-key"},
        ),
        client.delete(
            f"/v1/agents/{missing_agent_id}",
            headers={"Authorization": "Bearer test-key"},
        ),
    ]

    for resp in responses:
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert isinstance(detail, str)
        assert missing_agent_id in detail


@pytest.mark.asyncio
async def test_get_agent_daily_activity_admin_param_passing(monkeypatch):
    mock_prisma = AsyncMock()
    mock_prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(agent_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_agent_daily_activity(
        agent_ids="agent-1,agent-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_agent_ids="agent-3",
        user_api_key_dict=auth,
    )

    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyagentspend"
    assert kwargs["entity_id_field"] == "agent_id"
    assert kwargs["entity_id"] == ["agent-1", "agent-2"]
    assert kwargs["exclude_entity_ids"] == ["agent-3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5
    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_agent_daily_activity_with_agent_names(monkeypatch):
    mock_prisma = AsyncMock()
    mock_agent1 = MagicMock()
    mock_agent1.agent_id = "agent-1"
    mock_agent1.agent_name = "First Agent"
    mock_agent2 = MagicMock()
    mock_agent2.agent_id = "agent-2"
    mock_agent2.agent_name = "Second Agent"

    mock_prisma.db.litellm_agentstable.find_many = AsyncMock(
        return_value=[mock_agent1, mock_agent2]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(agent_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    await get_agent_daily_activity(
        agent_ids="agent-1,agent-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_agent_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_metadata_field"] == {
        "agent-1": {"agent_name": "First Agent"},
        "agent-2": {"agent_name": "Second Agent"},
    }
