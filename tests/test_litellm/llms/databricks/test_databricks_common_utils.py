import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.common_utils import DatabricksBase


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_databricks_credentials"
    ) as mock_get_credentials:
        try:
            databricks_base.databricks_validate_environment(
                api_key=None,
                api_base="my_api_base",
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )
        except Exception:
            pass
        mock_get_credentials.assert_called_once()


def test_databricks_validate_environment_oauth_m2m_from_litellm_params():
    """Test that client_id/client_secret from litellm_params are used for OAuth M2M auth."""
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_oauth_m2m_token", return_value="fake_token"
    ) as mock_get_token:
        api_base, headers = databricks_base.databricks_validate_environment(
            api_key=None,
            api_base="https://my-workspace.databricks.com/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            client_id="my-client-id",
            client_secret="my-client-secret",
        )

        mock_get_token.assert_called_once_with(
            "https://my-workspace.databricks.com/serving-endpoints",
            "my-client-id",
            "my-client-secret",
        )
        assert headers["Authorization"] == "Bearer fake_token"


def test_databricks_validate_environment_oauth_m2m_params_override_env():
    """Test that litellm_params client_id/client_secret take priority over env vars."""
    databricks_base = DatabricksBase()

    with patch.dict(
        os.environ,
        {
            "DATABRICKS_CLIENT_ID": "env-client-id",
            "DATABRICKS_CLIENT_SECRET": "env-client-secret",
        },
    ), patch.object(
        databricks_base, "_get_oauth_m2m_token", return_value="fake_token"
    ) as mock_get_token:
        databricks_base.databricks_validate_environment(
            api_key=None,
            api_base="https://my-workspace.databricks.com/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            client_id="param-client-id",
            client_secret="param-client-secret",
        )

        # litellm_params values should take priority over env vars
        mock_get_token.assert_called_once_with(
            "https://my-workspace.databricks.com/serving-endpoints",
            "param-client-id",
            "param-client-secret",
        )


def test_databricks_validate_environment_oauth_m2m_falls_back_to_env():
    """Test that env vars are used when litellm_params don't have client_id/client_secret."""
    databricks_base = DatabricksBase()

    with patch.dict(
        os.environ,
        {
            "DATABRICKS_CLIENT_ID": "env-client-id",
            "DATABRICKS_CLIENT_SECRET": "env-client-secret",
        },
    ), patch.object(
        databricks_base, "_get_oauth_m2m_token", return_value="fake_token"
    ) as mock_get_token:
        databricks_base.databricks_validate_environment(
            api_key=None,
            api_base="https://my-workspace.databricks.com/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
        )

        # Should fall back to env vars
        mock_get_token.assert_called_once_with(
            "https://my-workspace.databricks.com/serving-endpoints",
            "env-client-id",
            "env-client-secret",
        )
