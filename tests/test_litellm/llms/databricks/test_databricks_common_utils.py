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


def test_configure_user_agent_with_optional_params():
    """Test that optional_params sets the User-Agent env vars"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    optional_params = {
        "databricks_partner": "carto",
        "databricks_product": "agentic-gis",
        "databricks_product_version": "1.0.0",
    }

    databricks_base._configure_databricks_user_agent(
        optional_params=optional_params
    )

    # Verify optional_params values were used
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "carto"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "agentic-gis/1.0.0"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)


def test_configure_user_agent_env_vars_preserved():
    """Test that existing environment variables are not overridden"""
    databricks_base = DatabricksBase()

    # Set env vars first
    os.environ["DATABRICKS_SDK_UPSTREAM"] = "existing-partner"
    os.environ["DATABRICKS_SDK_UPSTREAM_VERSION"] = "existing-version"

    optional_params = {
        "databricks_partner": "carto",
        "databricks_product": "agentic-gis",
        "databricks_product_version": "1.0.0",
    }

    databricks_base._configure_databricks_user_agent(
        optional_params=optional_params
    )

    # Verify existing env vars were NOT overridden
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "existing-partner"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "existing-version"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)
