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

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_endpoints import (
    CreateGuardrailRequest,
    PatchGuardrailRequest,
    RegisterGuardrailRequest,
    UpdateGuardrailRequest,
    apply_guardrail,
    approve_guardrail_submission,
    create_guardrail,
    delete_guardrail,
    get_guardrail_info,
    get_guardrail_submission,
    list_guardrail_submissions,
    list_guardrails_v2,
    patch_guardrail,
    register_guardrail,
    reject_guardrail_submission,
    update_guardrail,
)

MOCK_ADMIN_USER = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
from litellm.proxy.guardrails.guardrail_registry import (
    IN_MEMORY_GUARDRAIL_HANDLER,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import (
    ApplyGuardrailRequest,
    BaseLitellmParams,
    Guardrail,
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
async def test_list_guardrails_v2_masks_sensitive_data_in_db_guardrails(mocker):
    """Test that sensitive litellm_params are masked for DB guardrails in list response"""
    db_guardrail_with_secrets = {
        "guardrail_id": "secret-db-guardrail",
        "guardrail_name": "DB Guardrail with Secrets",
        "litellm_params": {
            "guardrail": "azure/text_moderations",
            "mode": "pre_call",
            "api_key": "sk-1234567890abcdef",
            "api_base": "https://api.secret.example.com",
        },
        "guardrail_info": {"description": "Test guardrail"},
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }

    mock_prisma_client = mocker.Mock()
    mock_prisma_client.db = mocker.Mock()
    mock_prisma_client.db.litellm_guardrailstable = mocker.Mock()
    mock_prisma_client.db.litellm_guardrailstable.find_many = AsyncMock(
        return_value=[db_guardrail_with_secrets]
    )

    mock_in_memory_handler = mocker.Mock()
    mock_in_memory_handler.list_in_memory_guardrails.return_value = []

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    response = await list_guardrails_v2()

    assert len(response.guardrails) == 1
    guardrail = response.guardrails[0]
    litellm_params = guardrail.litellm_params
    if isinstance(litellm_params, dict):
        params = litellm_params
    else:
        params = litellm_params.model_dump() if hasattr(litellm_params, "model_dump") else dict(litellm_params)

    # Sensitive keys (containing "key", "secret", "token", etc.) should be masked
    assert params["api_key"] != "sk-1234567890abcdef"
    assert "****" in str(params["api_key"])
    # Non-sensitive keys should remain unchanged
    assert params["guardrail"] == "azure/text_moderations"
    assert params["mode"] == "pre_call"
    assert params["api_base"] == "https://api.secret.example.com"


@pytest.mark.asyncio
async def test_list_guardrails_v2_masks_sensitive_data_in_config_guardrails(mocker):
    """Test that sensitive litellm_params are masked for in-memory/config guardrails in list response"""
    config_guardrail_with_secrets = {
        "guardrail_id": "secret-config-guardrail",
        "guardrail_name": "Config Guardrail with Secrets",
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "during_call",
            "api_key": "my-secret-bedrock-key",
            "vertex_credentials": "{sensitive_creds}",
        },
        "guardrail_info": {"description": "Test guardrail from config"},
    }

    mock_prisma_client = mocker.Mock()
    mock_prisma_client.db = mocker.Mock()
    mock_prisma_client.db.litellm_guardrailstable = mocker.Mock()
    mock_prisma_client.db.litellm_guardrailstable.find_many = AsyncMock(
        return_value=[]
    )

    mock_in_memory_handler = mocker.Mock()
    mock_in_memory_handler.list_in_memory_guardrails.return_value = [
        config_guardrail_with_secrets
    ]

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_in_memory_handler,
    )

    response = await list_guardrails_v2()

    assert len(response.guardrails) == 1
    guardrail = response.guardrails[0]
    litellm_params = guardrail.litellm_params
    if isinstance(litellm_params, dict):
        params = litellm_params
    else:
        params = litellm_params.model_dump() if hasattr(litellm_params, "model_dump") else dict(litellm_params)

    # Sensitive keys should be masked
    assert params["api_key"] != "my-secret-bedrock-key"
    assert "****" in str(params["api_key"])
    assert params["vertex_credentials"] != "{sensitive_creds}"
    assert "****" in str(params["vertex_credentials"])
    # Non-sensitive keys should remain unchanged
    assert params["guardrail"] == "bedrock"
    assert params["mode"] == "during_call"


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

    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

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

    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

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

    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

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
    from unittest.mock import AsyncMock, Mock, patch

    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )
    
    guardrail_hook = BedrockGuardrail(
        guardrailIdentifier="test-guardrail-id",
        guardrailVersion="1"
    )
    
    guardrail_hook.async_handler = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": "NONE", "outputs": []}
    
    test_request_data = {
        "api_key": "test-api-key-789"
    }
    
    with patch.object(guardrail_hook.async_handler, "post", AsyncMock(return_value=mock_response)), \
         patch.object(guardrail_hook, "_load_credentials") as mock_load_creds, \
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
            await create_guardrail(MOCK_CREATE_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)

        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)

    else:
        result = await create_guardrail(MOCK_CREATE_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)
        
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
            await update_guardrail("test-guardrail-id", MOCK_UPDATE_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)

        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)

    else:
        result = await update_guardrail("test-guardrail-id", MOCK_UPDATE_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)
        
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
        mock_in_memory_handler.sync_guardrail_from_db = mocker.Mock()
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_guardrail_registry)
        mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)
        
    elif scenario == "success_sync_fails":
        mock_prisma_client = mocker.Mock()
        mock_in_memory_handler.sync_guardrail_from_db = mocker.Mock(side_effect=Exception("Sync failed"))
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
            await patch_guardrail("test-guardrail-id", MOCK_PATCH_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)

        if scenario == "database_failure":
            assert "Database error" in str(exc_info.value.detail)
        elif scenario == "no_prisma_client":
            assert "Prisma client not initialized" in str(exc_info.value.detail)

    else:
        result = await patch_guardrail("test-guardrail-id", MOCK_PATCH_REQUEST, user_api_key_dict=MOCK_ADMIN_USER)
        
        assert result["guardrail_id"] == expected_result
        assert result["guardrail_name"] == "Test DB Guardrail"
        
        mock_guardrail_registry.update_guardrail_in_db.assert_called_once()
        
        mock_in_memory_handler.sync_guardrail_from_db.assert_called_once_with(
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
            await delete_guardrail(guardrail_id=expected_result, user_api_key_dict=MOCK_ADMIN_USER)
    else:
        result = await delete_guardrail(guardrail_id=expected_result, user_api_key_dict=MOCK_ADMIN_USER)
        
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


@pytest.mark.asyncio
async def test_apply_guardrail_not_found(mocker):
    """
    Test apply_guardrail endpoint returns proper error when guardrail is not found.
    """
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth

    # Mock the GUARDRAIL_REGISTRY to return None (guardrail not found)
    mock_registry = mocker.Mock()
    mock_registry.get_initialized_guardrail_callback.return_value = None
    mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry)
    
    # Create request
    request = ApplyGuardrailRequest(
        guardrail_name="non-existent-guardrail",
        text="Test input text"
    )
    
    # Mock user auth
    mock_user_auth = UserAPIKeyAuth()
    
    # Call endpoint and expect ProxyException
    with pytest.raises(ProxyException) as exc_info:
        await apply_guardrail(request=request, user_api_key_dict=mock_user_auth)
    
    # Verify error details
    assert str(exc_info.value.code) == "404"
    assert "not found" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_apply_guardrail_execution_error(mocker):
    """
    Test apply_guardrail endpoint handles exceptions from guardrail execution properly.
    """
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth

    # Mock guardrail that raises an exception
    mock_guardrail = mocker.Mock()
    mock_guardrail.apply_guardrail = AsyncMock(
        side_effect=Exception("Bedrock guardrail failed: Violated guardrail policy")
    )
    
    # Mock the GUARDRAIL_REGISTRY
    mock_registry = mocker.Mock()
    mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail
    mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry)
    
    # Create request
    request = ApplyGuardrailRequest(
        guardrail_name="test-guardrail",
        text="Test input text with forbidden content"
    )
    
    # Mock user auth
    mock_user_auth = UserAPIKeyAuth()
    
    # Call endpoint and expect ProxyException
    with pytest.raises(ProxyException) as exc_info:
        await apply_guardrail(request=request, user_api_key_dict=mock_user_auth)
    
    # Verify error is properly handled
    assert "Bedrock guardrail failed" in str(exc_info.value.message)

@pytest.mark.asyncio
async def test_get_guardrail_info_endpoint_config_guardrail(mocker):
    """
    Test get_guardrail_info endpoint returns proper response when guardrail is found in config.
    """
    from litellm.proxy.guardrails.guardrail_endpoints import get_guardrail_info

    # Mock prisma_client to not be None (patch at the source where it's imported from)
    mock_prisma = mocker.Mock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    # Mock the GUARDRAIL_REGISTRY to return None from DB (so it checks config)
    mock_registry = mocker.Mock()
    mock_registry.get_guardrail_by_id_from_db = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry)

    # Mock IN_MEMORY_GUARDRAIL_HANDLER at its source to return config guardrail
    mock_in_memory_handler = mocker.Mock()
    mock_in_memory_handler.get_guardrail_by_id.return_value = MOCK_CONFIG_GUARDRAIL
    mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)

    # Mock _get_masked_values to return values as-is
    mocker.patch(
        "litellm.litellm_core_utils.litellm_logging._get_masked_values",
        side_effect=lambda x, **kwargs: x
    )

    # Call endpoint and expect GuardrailInfoResponse
    result = await get_guardrail_info(guardrail_id="test-config-guardrail")

    # Verify the response is of the correct type
    assert isinstance(result, GuardrailInfoResponse)
    assert result.guardrail_id == "test-config-guardrail"
    assert result.guardrail_name == "Test Config Guardrail"
    assert result.guardrail_definition_location == "config"

@pytest.mark.asyncio
async def test_get_guardrail_info_endpoint_db_guardrail(mocker):
    """
    Test get_guardrail_info endpoint returns proper response when guardrail is found in DB.
    """
    from litellm.proxy.guardrails.guardrail_endpoints import get_guardrail_info

    # Mock prisma_client to not be None (patch at the source where it's imported from)
    mock_prisma = mocker.Mock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    # Mock the GUARDRAIL_REGISTRY to return a guardrail from DB
    mock_registry = mocker.Mock()
    mock_registry.get_guardrail_by_id_from_db = AsyncMock(return_value=MOCK_DB_GUARDRAIL)
    mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", mock_registry)

    # Mock IN_MEMORY_GUARDRAIL_HANDLER to return None
    mock_in_memory_handler = mocker.Mock()
    mock_in_memory_handler.get_guardrail_by_id.return_value = None
    mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", mock_in_memory_handler)

    # Call endpoint and expect GuardrailInfoResponse
    result = await get_guardrail_info(guardrail_id="test-db-guardrail")

    # Verify the response is of the correct type
    assert isinstance(result, GuardrailInfoResponse)
    assert result.guardrail_id == "test-db-guardrail"
    assert result.guardrail_name == "Test DB Guardrail"
    assert result.guardrail_definition_location == "db"


# --- Team guardrail registration (register / submissions) ---

MOCK_REGISTER_REQUEST = RegisterGuardrailRequest(
    guardrail_name="team-prompt-guard",
    litellm_params={
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "api_base": "https://guardrails.example.com/validate",
    },
    guardrail_info={"description": "Team prompt injection detector"},
)


@pytest.mark.asyncio
async def test_register_guardrail_success(mocker):
    """Register creates a row with status pending_review and returns guardrail_id."""
    mock_prisma = mocker.Mock()
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=None)
    created_row = mocker.Mock(
        guardrail_id="reg-123",
        guardrail_name=MOCK_REGISTER_REQUEST.guardrail_name,
        status="pending_review",
        submitted_at=datetime.now(),
    )
    mock_prisma.db.litellm_guardrailstable.create = AsyncMock(return_value=created_row)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    user = UserAPIKeyAuth(user_id="u1", user_email="alice@co.com", team_id="team-1")
    result = await register_guardrail(MOCK_REGISTER_REQUEST, user)

    assert result.guardrail_id == "reg-123"
    assert result.guardrail_name == MOCK_REGISTER_REQUEST.guardrail_name
    assert result.status == "pending_review"
    mock_prisma.db.litellm_guardrailstable.create.assert_called_once()
    call_data = mock_prisma.db.litellm_guardrailstable.create.call_args[1]["data"]
    assert call_data["status"] == "pending_review"
    assert call_data["guardrail_name"] == MOCK_REGISTER_REQUEST.guardrail_name


@pytest.mark.asyncio
async def test_register_guardrail_rejects_non_generic_api(mocker):
    """Register returns 400 when litellm_params.guardrail is not generic_guardrail_api."""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mocker.Mock())
    req = RegisterGuardrailRequest(
        guardrail_name="other-guard",
        litellm_params={"guardrail": "bedrock", "mode": "pre_call", "api_base": "https://x.com"},
    )
    user = UserAPIKeyAuth(user_id="u1", user_email="a@b.com", team_id="team-1")

    with pytest.raises(HTTPException) as exc_info:
        await register_guardrail(req, user)
    assert exc_info.value.status_code == 400
    assert "generic_guardrail_api" in exc_info.value.detail


@pytest.mark.asyncio
async def test_register_guardrail_requires_team_id(mocker):
    """Register returns 400 when API key has no associated team_id."""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mocker.Mock())
    user = UserAPIKeyAuth(user_id="u1", user_email="a@b.com", team_id=None)

    with pytest.raises(HTTPException) as exc_info:
        await register_guardrail(MOCK_REGISTER_REQUEST, user)
    assert exc_info.value.status_code == 400
    assert "team" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_register_guardrail_duplicate_name(mocker):
    """Register returns 400 when guardrail_name already exists."""
    mock_prisma = mocker.Mock()
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(
        return_value={"guardrail_name": MOCK_REGISTER_REQUEST.guardrail_name}
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_id="u1", user_email="a@b.com", team_id="team-1")

    with pytest.raises(HTTPException) as exc_info:
        await register_guardrail(MOCK_REGISTER_REQUEST, user)
    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_guardrail_submissions_requires_admin(mocker):
    """List submissions returns 403 when user is not admin."""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mocker.Mock())
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    with pytest.raises(HTTPException) as exc_info:
        await list_guardrail_submissions(user_api_key_dict=user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_list_guardrail_submissions_success(mocker):
    """List submissions returns list and summary for admin."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(
        guardrail_id="sub-1",
        guardrail_name="pending-guard",
        status="pending_review",
        team_id="t1",
        litellm_params={"guardrail": "generic_guardrail_api", "api_base": "https://x.com"},
        guardrail_info={
            "description": "A guard",
            "submitted_by_user_id": "u1",
            "submitted_by_email": "alice@co.com",
        },
        submitted_at=datetime.now(),
        reviewed_at=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_prisma.db.litellm_guardrailstable.find_many = AsyncMock(return_value=[row])
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await list_guardrail_submissions(user_api_key_dict=user)

    assert len(result.submissions) == 1
    assert result.submissions[0].guardrail_id == "sub-1"
    assert result.submissions[0].status == "pending_review"
    assert result.submissions[0].team_guardrail is True  # team_id is set
    assert result.summary.total >= 1
    assert result.summary.pending_review >= 1


@pytest.mark.asyncio
async def test_list_guardrail_submissions_returns_only_team_guardrails(mocker):
    """List submissions only returns team guardrails (team_id not null)."""
    mock_prisma = mocker.Mock()
    find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_guardrailstable.find_many = find_many
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    await list_guardrail_submissions(user_api_key_dict=user)

    calls = find_many.call_args_list
    assert len(calls) >= 1
    first_where = calls[0].kwargs.get("where", {})
    assert first_where.get("team_id") == {"not": None}


@pytest.mark.asyncio
async def test_list_guardrail_submissions_team_id_filter(mocker):
    """List submissions with team_id filter returns only that team's guardrails."""
    mock_prisma = mocker.Mock()
    row_abc = mocker.Mock(
        guardrail_id="team-1",
        guardrail_name="team-guard",
        status="active",
        team_id="team-abc",
        litellm_params={},
        guardrail_info={},
        submitted_at=None,
        reviewed_at=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    row_other = mocker.Mock(
        guardrail_id="team-2",
        guardrail_name="other-guard",
        status="active",
        team_id="team-xyz",
        litellm_params={},
        guardrail_info={},
        submitted_at=None,
        reviewed_at=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    find_many = AsyncMock(return_value=[row_abc, row_other])
    mock_prisma.db.litellm_guardrailstable.find_many = find_many
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await list_guardrail_submissions(
        user_api_key_dict=user, team_id="team-abc"
    )

    assert len(result.submissions) == 1
    assert result.submissions[0].guardrail_id == "team-1"
    assert result.submissions[0].team_guardrail is True
    assert result.summary.total == 2  # summary counts all team guardrails


@pytest.mark.asyncio
async def test_get_guardrail_submission_not_found(mocker):
    """Get submission returns 404 when guardrail_id does not exist."""
    mock_prisma = mocker.Mock()
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await get_guardrail_submission("nonexistent-id", user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_approve_guardrail_submission_success(mocker):
    """Approve sets status to active and initializes guardrail in memory."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(
        guardrail_id="approve-me",
        guardrail_name="my-guard",
        status="pending_review",
        litellm_params={"guardrail": "generic_guardrail_api", "mode": "pre_call", "api_base": "https://g.com"},
        guardrail_info={},
    )
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=row)
    mock_prisma.db.litellm_guardrailstable.update = AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_handler = mocker.Mock()
    mock_handler.initialize_guardrail = mocker.Mock()
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_handler,
    )
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await approve_guardrail_submission("approve-me", user)

    assert result["status"] == "active"
    assert result["guardrail_id"] == "approve-me"
    mock_prisma.db.litellm_guardrailstable.update.assert_called_once()
    call_data = mock_prisma.db.litellm_guardrailstable.update.call_args[1]["data"]
    assert call_data["status"] == "active"


@pytest.mark.asyncio
async def test_approve_guardrail_submission_not_pending(mocker):
    """Approve returns 400 when status is not pending_review."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(guardrail_id="x", guardrail_name="y", status="active")
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=row)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await approve_guardrail_submission("x", user)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_reject_guardrail_submission_success(mocker):
    """Reject sets status to rejected."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(guardrail_id="rej-1", guardrail_name="r", status="pending_review")
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=row)
    mock_prisma.db.litellm_guardrailstable.update = AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await reject_guardrail_submission("rej-1", user)

    assert result["status"] == "rejected"
    mock_prisma.db.litellm_guardrailstable.update.assert_called_once()
    call_data = mock_prisma.db.litellm_guardrailstable.update.call_args[1]["data"]
    assert call_data["status"] == "rejected"


# --- Tests for review fixes ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base,expected_detail",
    [
        ("file:///etc/passwd", "http or https scheme"),
        ("ftp://internal.host/data", "http or https scheme"),
        ("javascript:alert(1)", "http or https scheme"),
        ("://missing-scheme", "http or https scheme"),
        ("https://", "valid hostname"),
    ],
    ids=[
        "file_scheme",
        "ftp_scheme",
        "javascript_scheme",
        "no_scheme",
        "no_hostname",
    ],
)
async def test_register_guardrail_rejects_bad_api_base(mocker, api_base, expected_detail):
    """Register returns 400 when api_base has invalid scheme or missing hostname."""
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mocker.Mock())
    req = RegisterGuardrailRequest(
        guardrail_name="bad-url-guard",
        litellm_params={
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": api_base,
        },
    )
    user = UserAPIKeyAuth(user_id="u1", user_email="a@b.com", team_id="team-1")

    with pytest.raises(HTTPException) as exc_info:
        await register_guardrail(req, user)
    assert exc_info.value.status_code == 400
    assert expected_detail in exc_info.value.detail


@pytest.mark.asyncio
async def test_register_guardrail_accepts_valid_https_url(mocker):
    """Register accepts valid https api_base URLs."""
    mock_prisma = mocker.Mock()
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=None)
    created_row = mocker.Mock(
        guardrail_id="valid-url-123",
        guardrail_name="valid-guard",
        status="pending_review",
        submitted_at=datetime.now(),
    )
    mock_prisma.db.litellm_guardrailstable.create = AsyncMock(return_value=created_row)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    req = RegisterGuardrailRequest(
        guardrail_name="valid-guard",
        litellm_params={
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": "https://guardrails.example.com/v1/check",
        },
    )
    user = UserAPIKeyAuth(user_id="u1", user_email="a@b.com", team_id="team-1")

    result = await register_guardrail(req, user)
    assert result.guardrail_id == "valid-url-123"
    assert result.status == "pending_review"


@pytest.mark.asyncio
async def test_approve_guardrail_init_failure_returns_warning(mocker):
    """Approve returns a warning field when in-memory initialization fails."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(
        guardrail_id="warn-me",
        guardrail_name="fragile-guard",
        status="pending_review",
        litellm_params={
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": "https://g.com",
        },
        guardrail_info={},
    )
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=row)
    mock_prisma.db.litellm_guardrailstable.update = AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mock_handler = mocker.Mock()
    mock_handler.initialize_guardrail = mocker.Mock(
        side_effect=Exception("missing dependency")
    )
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_handler,
    )
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await approve_guardrail_submission("warn-me", user)

    assert result["status"] == "active"
    assert "warning" in result
    assert "failed to initialize" in result["warning"].lower()
    assert "missing dependency" in result["warning"]


@pytest.mark.asyncio
async def test_approve_guardrail_no_warning_on_success(mocker):
    """Approve does NOT include a warning field when init succeeds."""
    mock_prisma = mocker.Mock()
    row = mocker.Mock(
        guardrail_id="ok-guard",
        guardrail_name="good-guard",
        status="pending_review",
        litellm_params={
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": "https://g.com",
        },
        guardrail_info={},
    )
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=row)
    mock_prisma.db.litellm_guardrailstable.update = AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mock_handler = mocker.Mock()
    mock_handler.initialize_guardrail = mocker.Mock()  # no exception
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_handler,
    )
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    result = await approve_guardrail_submission("ok-guard", user)

    assert result["status"] == "active"
    assert "warning" not in result


@pytest.mark.asyncio
async def test_list_submissions_single_db_query(mocker):
    """List submissions makes exactly one find_many call (no redundant query)."""
    mock_prisma = mocker.Mock()
    find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_guardrailstable.find_many = find_many
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    await list_guardrail_submissions(user_api_key_dict=user)

    assert find_many.call_count == 1


@pytest.mark.asyncio
async def test_list_submissions_summary_counts_unaffected_by_filters(mocker):
    """Summary counts reflect all team guardrails regardless of status filter."""
    mock_prisma = mocker.Mock()
    pending_row = mocker.Mock(
        guardrail_id="p1", guardrail_name="p", status="pending_review",
        team_id="t1", litellm_params={}, guardrail_info={},
        submitted_at=None, reviewed_at=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    active_row = mocker.Mock(
        guardrail_id="a1", guardrail_name="a", status="active",
        team_id="t1", litellm_params={}, guardrail_info={},
        submitted_at=None, reviewed_at=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    all_rows = [pending_row, active_row]
    mock_prisma.db.litellm_guardrailstable.find_many = AsyncMock(return_value=all_rows)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    # Filter to only pending, but summary should still show both
    result = await list_guardrail_submissions(status="pending_review", user_api_key_dict=user)

    assert len(result.submissions) == 1  # filtered
    assert result.summary.total == 2  # unfiltered
    assert result.summary.pending_review == 1
    assert result.summary.active == 1