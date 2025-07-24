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
    BaseLitellmParams,
    GuardrailInfoResponse,
    LitellmParams,
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
    assert isinstance(db_guardrail.litellm_params, BaseLitellmParams)

    # Check config guardrail
    config_guardrail = next(
        g for g in response.guardrails if g.guardrail_id == "test-config-guardrail"
    )
    assert config_guardrail.guardrail_name == "Test Config Guardrail"
    assert config_guardrail.guardrail_definition_location == "config"
    assert isinstance(config_guardrail.litellm_params, BaseLitellmParams)


@pytest.mark.asyncio
async def test_get_guardrail_info_from_db(mocker, mock_prisma_client):
    """Test getting guardrail info from DB"""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    response = await get_guardrail_info("test-db-guardrail")

    assert response.guardrail_id == "test-db-guardrail"
    assert response.guardrail_name == "Test DB Guardrail"
    assert isinstance(response.litellm_params, BaseLitellmParams)
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
    assert isinstance(response.litellm_params, BaseLitellmParams)
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


def test_get_provider_specific_params():
    """Test getting provider-specific parameters"""
    from litellm.proxy.guardrails.guardrail_endpoints import _get_fields_from_model
    from litellm.proxy.guardrails.guardrail_hooks.azure import (
        AzureContentSafetyTextModerationGuardrail,
    )

    config_model = AzureContentSafetyTextModerationGuardrail.get_config_model()
    if config_model is None:
        pytest.skip("Azure config model not available")

    fields = _get_fields_from_model(config_model)
    print("FIELDS", fields)

    # Test that we get the expected nested structure
    assert isinstance(fields, dict)

    # Check that we have the expected top-level fields
    assert "api_key" in fields
    assert "api_base" in fields
    assert "api_version" in fields
    assert "optional_params" in fields

    # Check the structure of a simple field
    assert (
        fields["api_key"]["description"]
        == "API key for the Azure Content Safety Prompt Shield guardrail"
    )
    assert fields["api_key"]["required"] == False
    assert fields["api_key"]["type"] == "string"  # Should be string, not None

    # Check the structure of the nested optional_params field
    assert fields["optional_params"]["type"] == "nested"
    assert fields["optional_params"]["required"] == True
    assert "fields" in fields["optional_params"]

    # Check nested fields within optional_params
    nested_fields = fields["optional_params"]["fields"]
    assert "severity_threshold" in nested_fields
    assert "severity_threshold_by_category" in nested_fields
    assert "categories" in nested_fields
    assert "blocklistNames" in nested_fields
    assert "haltOnBlocklistHit" in nested_fields
    assert "outputType" in nested_fields

    # Check structure of a nested field
    assert (
        nested_fields["severity_threshold"]["description"]
        == "Severity threshold for the Azure Content Safety Text Moderation guardrail across all categories"
    )
    assert nested_fields["severity_threshold"]["required"] == False
    assert (
        nested_fields["severity_threshold"]["type"] == "number"
    )  # Should be number, not None

    # Check other field types
    assert nested_fields["categories"]["type"] == "multiselect"
    assert nested_fields["blocklistNames"]["type"] == "array"
    assert nested_fields["haltOnBlocklistHit"]["type"] == "boolean"
    assert (
        nested_fields["outputType"]["type"] == "select"
    )  # Literal type should be select


def test_optional_params_not_returned_when_not_overridden():
    """Test that optional_params is not returned when the config model doesn't override it"""
    from typing import Optional

    from pydantic import BaseModel, Field

    from litellm.proxy.guardrails.guardrail_endpoints import _get_fields_from_model
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

    class TestGuardrailConfig(GuardrailConfigModel):
        api_key: Optional[str] = Field(
            default=None,
            description="Test API key",
        )
        api_base: Optional[str] = Field(
            default=None,
            description="Test API base",
        )

        @staticmethod
        def ui_friendly_name() -> str:
            return "Test Guardrail"

    # Get fields from the model
    fields = _get_fields_from_model(TestGuardrailConfig)
    print("FIELDS", fields)
    assert "optional_params" not in fields


def test_optional_params_returned_when_properly_overridden():
    """Test that optional_params IS returned when the config model properly overrides it"""
    from typing import Optional

    from pydantic import BaseModel, Field

    from litellm.proxy.guardrails.guardrail_endpoints import _get_fields_from_model
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

    # Create specific optional params model
    class SpecificOptionalParams(BaseModel):
        threshold: Optional[float] = Field(
            default=0.5, description="Detection threshold"
        )
        categories: Optional[List[str]] = Field(
            default=None, description="Categories to check"
        )

    # Create a config model that DOES override optional_params with a specific type
    class TestGuardrailConfigWithOptionalParams(
        GuardrailConfigModel[SpecificOptionalParams]
    ):
        api_key: Optional[str] = Field(
            default=None,
            description="Test API key",
        )

        @staticmethod
        def ui_friendly_name() -> str:
            return "Test Guardrail With Optional Params"

    # Get fields from the model
    fields = _get_fields_from_model(TestGuardrailConfigWithOptionalParams)

    print("FIELDS", fields)
    assert "optional_params" in fields
