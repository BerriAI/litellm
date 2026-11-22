import os
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.agentops.agentops import AgentOps, AgentOpsConfig
from litellm.llms.custom_httpx.http_handler import get_httpx_client


@pytest.fixture
def mock_auth_response():
    return {"token": "test_jwt_token", "project_id": "test_project_id"}


@pytest.fixture
def agentops_config():
    return AgentOpsConfig(
        endpoint="https://otlp.agentops.cloud/v1/traces",
        api_key="test_api_key",
        service_name="test_service",
        deployment_environment="test_env",
        auth_endpoint="https://api.agentops.ai/v3/auth/token",
    )


def test_agentops_config_from_env():
    """Test that AgentOpsConfig correctly reads from environment variables"""
    with patch.dict(
        os.environ,
        {
            "AGENTOPS_API_KEY": "test_key",
            "AGENTOPS_SERVICE_NAME": "test_service",
            "AGENTOPS_ENVIRONMENT": "test_env",
        },
    ):
        config = AgentOpsConfig.from_env()
        assert config.api_key == "test_key"
        assert config.service_name == "test_service"
        assert config.deployment_environment == "test_env"
        assert config.endpoint == "https://otlp.agentops.cloud/v1/traces"
        assert config.auth_endpoint == "https://api.agentops.ai/v3/auth/token"


def test_agentops_config_defaults():
    """Test that AgentOpsConfig uses correct default values"""
    config = AgentOpsConfig()
    assert config.service_name is None
    assert config.deployment_environment is None
    assert config.api_key is None
    assert config.endpoint == "https://otlp.agentops.cloud/v1/traces"
    assert config.auth_endpoint == "https://api.agentops.ai/v3/auth/token"


@patch("litellm.integrations.agentops.agentops.AgentOps._fetch_auth_token")
def test_fetch_auth_token_success(mock_fetch_auth_token, mock_auth_response):
    """Test successful JWT token fetch"""
    mock_fetch_auth_token.return_value = mock_auth_response

    config = AgentOpsConfig(api_key="test_key")
    agentops = AgentOps(config=config)

    mock_fetch_auth_token.assert_called_once_with(
        "test_key", "https://api.agentops.ai/v3/auth/token"
    )
    assert agentops.resource_attributes.get("project.id") == mock_auth_response.get(
        "project_id"
    )


@patch("litellm.integrations.agentops.agentops.AgentOps._fetch_auth_token")
def test_fetch_auth_token_failure(mock_fetch_auth_token):
    """Test failed JWT token fetch"""
    mock_fetch_auth_token.side_effect = Exception(
        "Failed to fetch auth token: Unauthorized"
    )

    config = AgentOpsConfig(api_key="test_key")
    agentops = AgentOps(config=config)

    mock_fetch_auth_token.assert_called_once()
    assert "project.id" not in agentops.resource_attributes


@patch("litellm.integrations.agentops.agentops.AgentOps._fetch_auth_token")
def test_agentops_initialization(
    mock_fetch_auth_token, agentops_config, mock_auth_response
):
    """Test AgentOps initialization with config"""
    mock_fetch_auth_token.return_value = mock_auth_response

    agentops = AgentOps(config=agentops_config)

    assert agentops.resource_attributes["service.name"] == "test_service"
    assert agentops.resource_attributes["deployment.environment"] == "test_env"
    assert agentops.resource_attributes["telemetry.sdk.name"] == "agentops"
    assert agentops.resource_attributes["project.id"] == "test_project_id"


def test_agentops_initialization_no_auth():
    """Test AgentOps initialization without authentication"""
    test_config = AgentOpsConfig(
        endpoint="https://otlp.agentops.cloud/v1/traces",
        api_key=None,  # No API key
        service_name="test_service",
        deployment_environment="test_env",
    )

    agentops = AgentOps(config=test_config)

    assert agentops.resource_attributes["service.name"] == "test_service"
    assert agentops.resource_attributes["deployment.environment"] == "test_env"
    assert agentops.resource_attributes["telemetry.sdk.name"] == "agentops"
    assert "project.id" not in agentops.resource_attributes


def test_auth_fetch_leaves_the_shared_sync_client_open_for_other_callers(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    auth_endpoint: Final = "https://agentops.example/v3/auth/token"
    respx_mock.post(auth_endpoint).respond(json={"token": "jwt", "project_id": "project-1"})
    client_another_caller_is_using: Final = get_httpx_client().client

    agentops: Final = AgentOps(config=AgentOpsConfig(api_key="key", auth_endpoint=auth_endpoint))

    assert agentops.resource_attributes["project.id"] == "project-1"
    assert not client_another_caller_is_using.is_closed
