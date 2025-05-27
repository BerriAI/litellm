import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_endpoints import (
    get_guardrail_info,
    list_guardrails_v2,
)
from litellm.proxy.guardrails.guardrail_registry import (
    IN_MEMORY_GUARDRAIL_HANDLER,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import (
    GuardrailInfoLiteLLMParamsResponse,
    GuardrailInfoResponse,
)

# Mock data for testing
MOCK_DB_GUARDRAIL = {
    "guardrail_id": "test-db-guardrail",
    "guardrail_name": "Test DB Guardrail",
    "litellm_params": {
        "guardrail": "test.guardrail",
        "mode": "pre_call",
    },
    "guardrail_info": {"description": "Test guardrail from DB"},
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
}

MOCK_CONFIG_GUARDRAIL = {
    "guardrail_id": "test-config-guardrail",
    "guardrail_name": "Test Config Guardrail",
    "litellm_params": {
        "guardrail": "custom_guardrail.myCustomGuardrail",
        "mode": "during_call",
    },
    "guardrail_info": {"description": "Test guardrail from config"},
}


@pytest.fixture
def mock_prisma_client(mocker):
    """Mock Prisma client for testing"""
    mock_client = mocker.Mock()
    # Create async mocks for the database methods
    mock_client.db = mocker.Mock()
    mock_client.db.litellm_guardrailstable = mocker.Mock()
    mock_client.db.litellm_guardrailstable.find_many = AsyncMock(
        return_value=[MOCK_DB_GUARDRAIL]
    )
    mock_client.db.litellm_guardrailstable.find_unique = AsyncMock(
        return_value=MOCK_DB_GUARDRAIL
    )
    return mock_client


@pytest.fixture
def mock_in_memory_handler(mocker):
    """Mock InMemoryGuardrailHandler for testing"""
    mock_handler = mocker.Mock(spec=InMemoryGuardrailHandler)
    mock_handler.list_in_memory_guardrails.return_value = [MOCK_CONFIG_GUARDRAIL]
    mock_handler.get_guardrail_by_id.return_value = MOCK_CONFIG_GUARDRAIL
    return mock_handler


@pytest.mark.asyncio
async def test_list_guardrails_v2_with_db_and_config(
    mocker, mock_prisma_client, mock_in_memory_handler
):
    """Test listing guardrails from both DB and config"""
    # Mock the prisma client
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    # Mock the in-memory handler
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    response = await list_guardrails_v2()

    assert len(response.guardrails) == 2

    # Check DB guardrail
    db_guardrail = next(
        g for g in response.guardrails if g.guardrail_id == "test-db-guardrail"
    )
    assert db_guardrail.guardrail_name == "Test DB Guardrail"
    assert db_guardrail.guardrail_definition_location == "db"
    assert isinstance(db_guardrail.litellm_params, GuardrailInfoLiteLLMParamsResponse)

    # Check config guardrail
    config_guardrail = next(
        g for g in response.guardrails if g.guardrail_id == "test-config-guardrail"
    )
    assert config_guardrail.guardrail_name == "Test Config Guardrail"
    assert config_guardrail.guardrail_definition_location == "config"
    assert isinstance(
        config_guardrail.litellm_params, GuardrailInfoLiteLLMParamsResponse
    )


@pytest.mark.asyncio
async def test_get_guardrail_info_from_db(mocker, mock_prisma_client):
    """Test getting guardrail info from DB"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    response = await get_guardrail_info("test-db-guardrail")

    assert response.guardrail_id == "test-db-guardrail"
    assert response.guardrail_name == "Test DB Guardrail"
    assert isinstance(response.litellm_params, GuardrailInfoLiteLLMParamsResponse)
    assert response.guardrail_info == {"description": "Test guardrail from DB"}


@pytest.mark.asyncio
async def test_get_guardrail_info_from_config(
    mocker, mock_prisma_client, mock_in_memory_handler
):
    """Test getting guardrail info from config when not found in DB"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    # Mock DB to return None
    mock_prisma_client.db.litellm_guardrailstable.find_unique = AsyncMock(
        return_value=None
    )

    response = await get_guardrail_info("test-config-guardrail")

    assert response.guardrail_id == "test-config-guardrail"
    assert response.guardrail_name == "Test Config Guardrail"
    assert isinstance(response.litellm_params, GuardrailInfoLiteLLMParamsResponse)
    assert response.guardrail_info == {"description": "Test guardrail from config"}


@pytest.mark.asyncio
async def test_get_guardrail_info_not_found(
    mocker, mock_prisma_client, mock_in_memory_handler
):
    """Test getting guardrail info when not found in either DB or config"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    # Mock both DB and in-memory handler to return None
    mock_prisma_client.db.litellm_guardrailstable.find_unique = AsyncMock(
        return_value=None
    )
    mock_in_memory_handler.get_guardrail_by_id.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_guardrail_info("non-existent-guardrail")

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail)
