import pytest
import sys
import os
from unittest.mock import AsyncMock
from fastapi import HTTPException

# Add repo root to path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy.guardrails.guardrail_endpoints import (
    create_guardrail,
    update_guardrail,
    list_guardrails,
)
from litellm.types.guardrails import (
    CreateGuardrailRequest,
    UpdateGuardrailRequest,
)

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# Fixtures
@pytest.fixture
def mock_team_user_auth():
    return UserAPIKeyAuth(user_role=LitellmUserRoles.TEAM, team_id="team-123")


@pytest.fixture
def mock_other_team_user_auth():
    return UserAPIKeyAuth(user_role=LitellmUserRoles.TEAM, team_id="team-456")


@pytest.fixture
def mock_prisma_client(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_in_memory_handler(mocker):
    mock = mocker.Mock()
    mock.get_guardrail_by_id.return_value = None
    mock.list_in_memory_guardrails.return_value = []
    return mock


@pytest.mark.asyncio
async def test_team_list_guardrails_v2_isolation(
    mocker,
    mock_prisma_client,
    mock_in_memory_handler,
    mock_team_user_auth,
    mock_other_team_user_auth,
):
    """Test that teams only see their own guardrails in list endpoint"""
    # Setup mocks
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    async def mock_get_all_guardrails(prisma_client=None, team_id=None):
        all_guardrails = [
            {
                "guardrail_id": "g1",
                "guardrail_name": "G1",
                "team_id": "team-123",
                "guardrail_config": {},
                "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
            },
            {
                "guardrail_id": "g2",
                "guardrail_name": "G2",
                "team_id": "team-456",
                "guardrail_config": {},
                "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
            },
            {
                "guardrail_id": "g3",
                "guardrail_name": "G3",
                "team_id": None,
                "guardrail_config": {},
                "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
            },
        ]
        if team_id:
            return [g for g in all_guardrails if g["team_id"] == team_id]
        return all_guardrails

    mock_registry = mocker.Mock()
    mock_registry.get_all_guardrails_from_db = AsyncMock(
        side_effect=mock_get_all_guardrails
    )
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry
    )

    # Test for Team 123
    response_123 = await list_guardrails(user_api_key_dict=mock_team_user_auth)
    assert len(response_123.guardrails) == 1
    assert response_123.guardrails[0].guardrail_id == "g1"

    # Test for Team 456
    response_456 = await list_guardrails(user_api_key_dict=mock_other_team_user_auth)
    assert len(response_456.guardrails) == 1
    assert response_456.guardrails[0].guardrail_id == "g2"


@pytest.mark.asyncio
async def test_team_create_guardrail_sets_team_id(
    mocker, mock_prisma_client, mock_team_user_auth
):
    """Test that creating a guardrail as a team user sets the team_id"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_registry = mocker.Mock()
    mock_registry.get_guardrail_by_name_from_db = AsyncMock(return_value=None)

    async def mock_add_guardrail(guardrail, team_id=None, **kwargs):
        return {
            "guardrail_id": "new-g",
            "guardrail_name": guardrail["guardrail_name"],
            "team_id": team_id,
            "created_at": "2024-01-01T00:00:00.000Z",
            "updated_at": "2024-01-01T00:00:00.000Z",
        }

    mock_registry.add_guardrail_to_db = AsyncMock(side_effect=mock_add_guardrail)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry
    )

    # Mock initialize_guardrail which is called after success
    # NOTE: The endpoint calls initialize_guardrail on IN_MEMORY_GUARDRAIL_HANDLER imported from guardrail_registry
    mock_in_memory = mocker.Mock()
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory,
    )

    request = CreateGuardrailRequest(
        guardrail={
            "guardrail_name": "New Team Guardrail",
            "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
            "guardrail_info": {},
        }
    )

    response = await create_guardrail(
        request=request, user_api_key_dict=mock_team_user_auth
    )

    mock_registry.add_guardrail_to_db.assert_called_once()
    call_kwargs = mock_registry.add_guardrail_to_db.call_args[1]
    assert call_kwargs["team_id"] == "team-123"
    assert response["team_id"] == "team-123"


@pytest.mark.asyncio
async def test_team_update_other_team_guardrail_fails(
    mocker, mock_prisma_client, mock_team_user_auth
):
    """Test that a team user cannot update another team's guardrail"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_registry = mocker.Mock()

    async def mock_get_by_id(guardrail_id, **kwargs):
        g = {
            "guardrail_id": "g2",
            "guardrail_name": "G2",
            "team_id": "team-456",
            "guardrail_config": {},
            "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
        }
        if g["guardrail_id"] == guardrail_id:
            return g
        return None

    mock_registry.get_guardrail_by_id_from_db = AsyncMock(side_effect=mock_get_by_id)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry
    )

    request = UpdateGuardrailRequest(
        guardrail={
            "guardrail_name": "Updated Name",
            "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"},
        }
    )

    # Team 123 tries to update G2 (owned by 456)
    with pytest.raises(HTTPException) as excinfo:
        await update_guardrail(
            guardrail_id="g2", request=request, user_api_key_dict=mock_team_user_auth
        )

    assert excinfo.value.status_code == 403
