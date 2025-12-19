import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock infisical module before importing InfisicalSecretManager
# This ensures we don't need the actual package installed for unit testing the logic
mock_infisical = MagicMock()
sys.modules["infisical"] = mock_infisical

from litellm.secret_managers.infisical_secret_manager import InfisicalSecretManager

@pytest.fixture
def mock_infisical_client():
    # Reset the mock for each test
    mock_infisical.InfisicalClient.reset_mock()
    return mock_infisical.InfisicalClient.return_value

def test_infisical_secret_manager_init():
    """
    Test initialization of InfisicalSecretManager
    """
    manager = InfisicalSecretManager(
        infisical_client_id="test_id",
        infisical_client_secret="test_secret",
        infisical_site_url="https://test.infisical.com"
    )
    assert manager.client is not None
    mock_infisical.InfisicalClient.assert_called()

def test_read_secret(mock_infisical_client):
    """
    Test reading a secret
    """
    mock_infisical_client.getSecret.return_value.secretValue = "test_value"
    
    manager = InfisicalSecretManager(
        infisical_client_id="test_id",
        infisical_client_secret="test_secret"
    )
    
    val = manager.sync_read_secret("test_key")
    assert val == "test_value"
    mock_infisical_client.getSecret.assert_called_once()

@pytest.mark.asyncio
async def test_async_write_secret(mock_infisical_client):
    """
    Test writing a secret
    """
    mock_infisical_client.createSecret.return_value.secretName = "test_key"
    mock_infisical_client.createSecret.return_value.version = 1
    
    manager = InfisicalSecretManager(
        infisical_client_id="test_id",
        infisical_client_secret="test_secret"
    )
    
    res = await manager.async_write_secret("test_key", "test_value")
    assert res["name"] == "test_key"
    assert res["version"] == 1
    mock_infisical_client.createSecret.assert_called_once()

@pytest.mark.asyncio
async def test_async_delete_secret(mock_infisical_client):
    """
    Test deleting a secret
    """
    manager = InfisicalSecretManager(
        infisical_client_id="test_id",
        infisical_client_secret="test_secret"
    )
    
    res = await manager.async_delete_secret("test_key")
    assert res["deleted"] is True
    mock_infisical_client.deleteSecret.assert_called_once()
