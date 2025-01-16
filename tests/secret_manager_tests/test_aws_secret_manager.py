# What is this?

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils


load_dotenv()
import io

import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

print("Python Path:", sys.path)
print("Current Working Directory:", os.getcwd())


from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import uuid
import json
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2


def check_aws_credentials():
    """Helper function to check if AWS credentials are set"""
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Missing required AWS credentials: {', '.join(missing_vars)}")


@pytest.mark.asyncio
async def test_write_and_read_simple_secret():
    """Test writing and reading a simple string secret"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    test_secret_name = f"litellm_test_{uuid.uuid4().hex[:8]}"
    test_secret_value = "test_value_123"

    try:
        # Write secret
        write_response = await secret_manager.async_write_secret(
            secret_name=test_secret_name,
            secret_value=test_secret_value,
            description="LiteLLM Test Secret",
        )

        print("Write Response:", write_response)

        assert write_response is not None
        assert "ARN" in write_response
        assert "Name" in write_response
        assert write_response["Name"] == test_secret_name

        # Read secret back
        read_value = await secret_manager.async_read_secret(
            secret_name=test_secret_name
        )

        print("Read Value:", read_value)

        assert read_value == test_secret_value
    finally:
        # Cleanup: Delete the secret
        delete_response = await secret_manager.async_delete_secret(
            secret_name=test_secret_name
        )
        print("Delete Response:", delete_response)
        assert delete_response is not None


@pytest.mark.asyncio
async def test_write_and_read_json_secret():
    """Test writing and reading a JSON structured secret"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    test_secret_name = f"litellm_test_{uuid.uuid4().hex[:8]}_json"
    test_secret_value = {
        "api_key": "test_key",
        "model": "gpt-4",
        "temperature": 0.7,
        "metadata": {"team": "ml", "project": "litellm"},
    }

    try:
        # Write JSON secret
        write_response = await secret_manager.async_write_secret(
            secret_name=test_secret_name,
            secret_value=json.dumps(test_secret_value),
            description="LiteLLM JSON Test Secret",
        )

        print("Write Response:", write_response)

        # Read and parse JSON secret
        read_value = await secret_manager.async_read_secret(
            secret_name=test_secret_name
        )
        parsed_value = json.loads(read_value)

        print("Read Value:", read_value)

        assert parsed_value == test_secret_value
        assert parsed_value["api_key"] == "test_key"
        assert parsed_value["metadata"]["team"] == "ml"
    finally:
        # Cleanup: Delete the secret
        delete_response = await secret_manager.async_delete_secret(
            secret_name=test_secret_name
        )
        print("Delete Response:", delete_response)
        assert delete_response is not None


@pytest.mark.asyncio
async def test_read_nonexistent_secret():
    """Test reading a secret that doesn't exist"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    nonexistent_secret = f"litellm_nonexistent_{uuid.uuid4().hex}"

    response = await secret_manager.async_read_secret(secret_name=nonexistent_secret)

    assert response is None
