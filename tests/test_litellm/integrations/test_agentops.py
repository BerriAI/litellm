import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.integrations.agentops.agentops import AgentOps, AgentOpsConfig


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
