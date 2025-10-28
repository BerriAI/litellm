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
    CreateGuardrailRequest,
    create_guardrail,
    UpdateGuardrailRequest,
    update_guardrail,
    PatchGuardrailRequest,
    patch_guardrail,
    delete_guardrail,
)
from litellm.proxy.guardrails.guardrail_registry import (
    IN_MEMORY_GUARDRAIL_HANDLER,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import (
    BaseLitellmParams,
    GuardrailInfoResponse,
    LitellmParams,
    Guardrail,
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

MOCK_GUARDRAIL = Guardrail(
    guardrail_name=MOCK_CONFIG_GUARDRAIL["guardrail_name"],
    litellm_params=LitellmParams(**MOCK_CONFIG_GUARDRAIL["litellm_params"]),
    guardrail_info=MOCK_CONFIG_GUARDRAIL["guardrail_info"]
)

MOCK_CREATE_REQUEST = CreateGuardrailRequest(guardrail=MOCK_GUARDRAIL)
MOCK_UPDATE_REQUEST = UpdateGuardrailRequest(guardrail=MOCK_GUARDRAIL)
MOCK_PATCH_REQUEST = PatchGuardrailRequest(
    guardrail_name="Updated Test Guardrail",
    litellm_params={"guardrail": "updated.guardrail", "mode": "post_call"},
    guardrail_info={"description": "Updated test guardrail"}
)


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
    mock_handler.initialize_guardrail = mocker.Mock()
    mock_handler.update_in_memory_guardrail = mocker.Mock()
    mock_handler.delete_in_memory_guardrail = mocker.Mock()
    return mock_handler

@pytest.fixture
def mock_guardrail_registry(mocker):
    """Mock GuardrailRegistry for testing"""
    mock_registry = mocker.Mock()
    mock_registry.add_guardrail_to_db = AsyncMock(return_value={
        **MOCK_DB_GUARDRAIL,
        "guardrail_id": "new-test-guardrail-id"
    })
    mock_registry.delete_guardrail_from_db = AsyncMock(return_value=MOCK_DB_GUARDRAIL)
    mock_registry.get_guardrail_by_id_from_db = AsyncMock(return_value=MOCK_DB_GUARDRAIL)
    mock_registry.update_guardrail_in_db = AsyncMock(return_value=MOCK_DB_GUARDRAIL)
    return mock_registry

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


@pytest.mark.asyncio
async def test_bedrock_guardrail_prepare_request_with_api_key():
    """Test _prepare_request method uses Bearer token when api_key is provided in data"""
    from unittest.mock import Mock, patch
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    
    # Setup guardrail hook
    guardrail_hook = BedrockGuardrail(
        guardrailIdentifier="test-guardrail-id",
        guardrailVersion="1"
    )
    mock_credentials = Mock()
    test_data = {
        "source": "INPUT",
        "content": [{"text": {"text": "test content"}}]
    }
    
    prepared_request = guardrail_hook._prepare_request(
        credentials=mock_credentials,
        data=test_data,
        optional_params={},
        aws_region_name="us-east-1",
        api_key="test-bearer-token-123"
    )
    
    # Verify Bearer token is used in Authorization header
    assert "Authorization" in prepared_request.headers
    assert prepared_request.headers["Authorization"] == "Bearer test-bearer-token-123"
    
    # Verify URL is correct
    expected_url = "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail/test-guardrail-id/version/1/apply"
    assert prepared_request.url == expected_url


@pytest.mark.asyncio
async def test_bedrock_guardrail_prepare_request_without_api_key():
    """Test _prepare_request method falls back to SigV4 when no api_key is provided"""
    from unittest.mock import Mock, patch
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    
    # Setup guardrail hook
    guardrail_hook = BedrockGuardrail(
        guardrailIdentifier="test-guardrail-id",
        guardrailVersion="1"
    )
    
    # Mock credentials
    mock_credentials = Mock()
    
    # Test data without api_key
    test_data = {
        "source": "INPUT",
        "content": [{"text": {"text": "test content"}}]
    }
    
    with patch("litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.get_secret_str") as mock_get_secret, \
         patch("botocore.auth.SigV4Auth") as mock_sigv4_auth, \
         patch("botocore.awsrequest.AWSRequest") as mock_aws_request:
        
        # Mock no AWS_BEARER_TOKEN_BEDROCK
        mock_get_secret.return_value = None
        
        # Mock SigV4Auth
        mock_sigv4_instance = Mock()
        mock_sigv4_auth.return_value = mock_sigv4_instance
        
        # Mock AWSRequest
        mock_request_instance = Mock()
        mock_request_instance.prepare.return_value = Mock()
        mock_aws_request.return_value = mock_request_instance
        
        # Call _prepare_request
        prepared_request = guardrail_hook._prepare_request(
            credentials=mock_credentials,
            data=test_data,
            optional_params={},
            aws_region_name="us-east-1"
        )
        
        # Verify SigV4 auth was used
        mock_sigv4_auth.assert_called_once_with(mock_credentials, "bedrock", "us-east-1")
        mock_sigv4_instance.add_auth.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_guardrail_prepare_request_with_bearer_token_env():
    """Test _prepare_request method uses Bearer token from environment when available"""
    from unittest.mock import Mock, patch
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    
    # Setup guardrail hook
    guardrail_hook = BedrockGuardrail(
        guardrailIdentifier="test-guardrail-id",
        guardrailVersion="1"
    )
    
    # Mock credentials
    mock_credentials = Mock()
    
    # Test data without api_key
    test_data = {
        "source": "INPUT",
        "content": [{"text": {"text": "test content"}}]
    }
    
    with patch("litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.get_secret_str") as mock_get_secret, \
         patch("botocore.awsrequest.AWSRequest") as mock_aws_request:
        
        mock_get_secret.return_value = "env-bearer-token-456"
        mock_request_instance = Mock()
        mock_request_instance.prepare.return_value = Mock()
        mock_aws_request.return_value = mock_request_instance
        
        prepared_request = guardrail_hook._prepare_request(
            credentials=mock_credentials,
            data=test_data,
            optional_params={},
            aws_region_name="us-east-1"
        )
        
        # Verify Bearer token from environment is used
        mock_aws_request.assert_called_once()
        call_args = mock_aws_request.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer env-bearer-token-456"


@pytest.mark.asyncio
async def test_bedrock_guardrail_make_api_request_passes_api_key():
    """Test make_bedrock_api_request method correctly passes api_key from request_data"""
    from unittest.mock import Mock, patch, AsyncMock
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    
    guardrail_hook = BedrockGuardrail(
        guardrailIdentifier="test-guardrail-id",
        guardrailVersion="1"
    )
    
    guardrail_hook.async_handler = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": "NONE", "outputs": []}
    guardrail_hook.async_handler.post = AsyncMock(return_value=mock_response)
    
    test_request_data = {
        "api_key": "test-api-key-789"
    }
    
    with patch.object(guardrail_hook, "_load_credentials") as mock_load_creds, \
         patch.object(guardrail_hook, "convert_to_bedrock_format") as mock_convert, \
         patch.object(guardrail_hook, "get_guardrail_dynamic_request_body_params") as mock_get_params, \
         patch.object(guardrail_hook, "add_standard_logging_guardrail_information_to_request_data"), \
         patch("botocore.awsrequest.AWSRequest") as mock_aws_request:
        
        mock_load_creds.return_value = (Mock(), "us-east-1")
        mock_convert.return_value = {"source": "INPUT", "content": []}
        mock_get_params.return_value = {}
        
        mock_request_instance = Mock()
        mock_request_instance.url = "test-url"
        mock_request_instance.body = b"test-body"
        mock_request_instance.headers = {"Content-Type": "application/json", "Authorization": "Bearer test-api-key-789"}
        mock_request_instance.prepare.return_value = Mock()
        mock_aws_request.return_value = mock_request_instance
        
        await guardrail_hook.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "test"}],
            request_data=test_request_data
        )
        
        # Verify _prepare_request was invoked and used the api_key
        mock_aws_request.assert_called_once()
        call_args = mock_aws_request.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-api-key-789"


@pytest.mark.parametrize("scenario,expected_result,expected_exception", [
    (
        "success_with_sync",
        "new-test-guardrail-id",
        None
    ),
    (
        "success_sync_fails", 
        "new-test-guardrail-id",
        None
    ),
    (
        "database_failure",
        None,
        HTTPException
    ),
    (
        "no_prisma_client",
        None,
        HTTPException
    ),
], ids=[
    "success_with_immediate_sync",
    "success_but_sync_fails",
    "database_error",
    "missing_prisma_client"
])
@pytest.mark.asyncio
async def test_create_guardrail_endpoint(
    scenario, expected_result, expected_exception,
    mocker, mock_guardrail_registry, mock_in_memory_handler
):
    """Test create_guardrail endpoint with different scenarios"""
    
    # Configure mocks based on scenario
    mock_logger = None
    if scenario == "success_with_sync":
        mock_prisma_client = mocker.Mock()
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "success_sync_fails":
        mock_prisma_client = mocker.Mock()
        mock_in_memory_handler.initialize_guardrail.side_effect = Exception("Sync failed")
        mock_logger = mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.verbose_proxy_logger")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "database_failure":
        mock_prisma_client = mocker.Mock()
        mock_guardrail_registry.add_guardrail_to_db.side_effect = Exception("Database error")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)  
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        
    elif scenario == "no_prisma_client":
        mocker.patch("litellm.proxy.proxy_server.prisma_client", None)
    
    # Run the test
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            await create_guardrail(MOCK_CREATE_REQUEST)
            
        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)
            
    else:
        result = await create_guardrail(MOCK_CREATE_REQUEST)
        
        assert result["guardrail_id"] == expected_result
        assert result["guardrail_name"] == "Test DB Guardrail"
        
        mock_guardrail_registry.add_guardrail_to_db.assert_called_once_with(
            guardrail=MOCK_CREATE_REQUEST.guardrail,
            prisma_client=mocker.ANY
        )
        
        mock_in_memory_handler.initialize_guardrail.assert_called_once()
        
        if scenario == "success_sync_fails":
            assert mock_logger is not None
            mock_logger.warning.assert_called_once()
            assert "Failed to initialize guardrail" in str(mock_logger.warning.call_args)

@pytest.mark.parametrize("scenario,expected_result,expected_exception", [
    (
        "success_with_sync",
        "test-db-guardrail",
        None
    ),
    (
        "success_sync_fails", 
        "test-db-guardrail",
        None
    ),
    (
        "database_failure",
        None,
        HTTPException
    ),
    (
        "no_prisma_client",
        None,
        HTTPException
    ),
], ids=[
    "success_with_immediate_sync",
    "success_but_sync_fails",
    "database_error",
    "missing_prisma_client"
])
@pytest.mark.asyncio
async def test_update_guardrail_endpoint(
    scenario, expected_result, expected_exception,
    mocker, mock_guardrail_registry, mock_in_memory_handler
):
    """Test update_guardrail endpoint with different scenarios"""
    
    # Configure mocks based on scenario
    mock_logger = None
    if scenario == "success_with_sync":
        mock_prisma_client = mocker.Mock()
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "success_sync_fails":
        mock_prisma_client = mocker.Mock()
        mock_in_memory_handler.update_in_memory_guardrail.side_effect = Exception("Sync failed")
        mock_logger = mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.verbose_proxy_logger")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "database_failure":
        mock_prisma_client = mocker.Mock()
        mock_guardrail_registry.update_guardrail_in_db.side_effect = Exception("Database error")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)  
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        
    elif scenario == "no_prisma_client":
        mocker.patch("litellm.proxy.proxy_server.prisma_client", None)
    
    # Run the test
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            await update_guardrail("test-guardrail-id", MOCK_UPDATE_REQUEST)
            
        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)
            
    else:
        result = await update_guardrail("test-guardrail-id", MOCK_UPDATE_REQUEST)
        
        assert result["guardrail_id"] == expected_result
        assert result["guardrail_name"] == "Test DB Guardrail"
        
        mock_guardrail_registry.update_guardrail_in_db.assert_called_once_with(
            guardrail_id="test-guardrail-id",
            guardrail=MOCK_UPDATE_REQUEST.guardrail,
            prisma_client=mocker.ANY
        )
        
        mock_in_memory_handler.update_in_memory_guardrail.assert_called_once_with(
            guardrail_id="test-guardrail-id",
            guardrail=mocker.ANY
        )
        
        if scenario == "success_sync_fails":
            assert mock_logger is not None
            mock_logger.warning.assert_called_once()
            assert "Failed to update" in str(mock_logger.warning.call_args)

@pytest.mark.parametrize("scenario,expected_result,expected_exception", [
    (
        "success_with_sync",
        "test-db-guardrail",
        None
    ),
    (
        "success_sync_fails", 
        "test-db-guardrail",
        None
    ),
    (
        "database_failure",
        None,
        HTTPException
    ),
    (
        "no_prisma_client",
        None,
        HTTPException
    ),
], ids=[
    "success_with_immediate_sync",
    "success_but_sync_fails",
    "database_error",
    "missing_prisma_client"
])
@pytest.mark.asyncio
async def test_patch_guardrail_endpoint(
    scenario, expected_result, expected_exception,
    mocker, mock_guardrail_registry, mock_in_memory_handler
):
    """Test patch_guardrail endpoint with different scenarios"""
    
    # Configure mocks based on scenario
    mock_logger = None
    if scenario == "success_with_sync":
        mock_prisma_client = mocker.Mock()
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "success_sync_fails":
        mock_prisma_client = mocker.Mock()
        mock_in_memory_handler.update_in_memory_guardrail.side_effect = Exception("Sync failed")
        mock_logger = mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.verbose_proxy_logger")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "database_failure":
        mock_prisma_client = mocker.Mock()
        mock_guardrail_registry.update_guardrail_in_db.side_effect = Exception("Database error")
        
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)  
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        
    elif scenario == "no_prisma_client":
        mocker.patch("litellm.proxy.proxy_server.prisma_client", None)
    
    # Run the test
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            await patch_guardrail("test-guardrail-id", MOCK_PATCH_REQUEST)
            
        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)
            
    else:
        result = await patch_guardrail("test-guardrail-id", MOCK_PATCH_REQUEST)
        
        assert result["guardrail_id"] == expected_result
        assert result["guardrail_name"] == "Test DB Guardrail"
        
        mock_guardrail_registry.update_guardrail_in_db.assert_called_once()
        
        mock_in_memory_handler.update_in_memory_guardrail.assert_called_once_with(
            guardrail_id="test-guardrail-id",
            guardrail=mocker.ANY
        )
        
        if scenario == "success_sync_fails":
            assert mock_logger is not None
            mock_logger.warning.assert_called_once()
            assert "Failed to update" in str(mock_logger.warning.call_args)

@pytest.mark.parametrize("scenario,expected_result,expected_exception", [
    (
        "success_with_sync",
        "test-db-guardrail",
        None
    ),
    (
        "success_sync_fails", 
        "test-db-guardrail",
        None
    ),
], ids=[
    "success_with_immediate_sync",
    "success_but_sync_fails"
])
@pytest.mark.asyncio
async def test_delete_guardrail_endpoint(
    scenario, expected_result, expected_exception,
    mocker, mock_guardrail_registry, mock_in_memory_handler
):
    """Test delete_guardrail endpoint with different scenarios"""
    
    # Configure mocks based on scenario
    mock_prisma_client = mocker.Mock()
    mock_logger = None
    
    if scenario == "success_with_sync":
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "success_sync_fails":
        mock_in_memory_handler.delete_in_memory_guardrail.side_effect = Exception("Sync failed")
        mock_logger = mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.verbose_proxy_logger")
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
    
    if expected_exception:
        with pytest.raises(expected_exception):
            await delete_guardrail(guardrail_id=expected_result)
    else:
        result = await delete_guardrail(guardrail_id=expected_result)
        
        assert result == MOCK_DB_GUARDRAIL
        
        mock_guardrail_registry.get_guardrail_by_id_from_db.assert_called_once_with(
            guardrail_id=expected_result, 
            prisma_client=mock_prisma_client
        )
        mock_guardrail_registry.delete_guardrail_from_db.assert_called_once_with(
            guardrail_id=expected_result,
            prisma_client=mock_prisma_client
        )
        
        mock_in_memory_handler.delete_in_memory_guardrail.assert_called_once_with(
            guardrail_id=expected_result
        )
        
        if scenario == "success_sync_fails":
            assert mock_logger is not None
            mock_logger.warning.assert_called_once()
            assert "Failed to remove guardrail" in str(mock_logger.warning.call_args)