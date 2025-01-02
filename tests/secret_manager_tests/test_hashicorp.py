import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()
import os
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch, MagicMock
import logging
from litellm._logging import verbose_logger

verbose_logger.setLevel(logging.DEBUG)

from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager

hashicorp_secret_manager = HashicorpSecretManager()


mock_vault_response = {
    "request_id": "80fafb6a-e96a-4c5b-29fa-ff505ac72201",
    "lease_id": "",
    "renewable": False,
    "lease_duration": 0,
    "data": {
        "data": {"key": "value-mock"},
        "metadata": {
            "created_time": "2025-01-01T22:13:50.93942388Z",
            "custom_metadata": None,
            "deletion_time": "",
            "destroyed": False,
            "version": 1,
        },
    },
    "wrap_info": None,
    "warnings": None,
    "auth": None,
    "mount_type": "kv",
}


def test_hashicorp_secret_manager_get_secret():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get") as mock_get:
        # Configure the mock response using MagicMock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_vault_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test the secret manager
        secret = hashicorp_secret_manager.read_secret("sample-secret-mock")
        assert secret == "value-mock"

        # Verify the request was made with correct parameters
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "sample-secret-mock" in called_url

        assert (
            called_url
            == "https://test-cluster-public-vault-0f98180c.e98296b2.z1.hashicorp.cloud:8200/v1/admin/secret/data/sample-secret-mock"
        )
        assert "X-Vault-Token" in mock_get.call_args.kwargs["headers"]
