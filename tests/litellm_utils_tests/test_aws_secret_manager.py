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
from litellm._uuid import uuid
import json
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.secret_managers.main import KeyManagementSettings


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


@pytest.mark.asyncio
async def test_primary_secret_functionality():
    """Test storing and retrieving secrets from a primary secret"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    primary_secret_name = f"litellm_test_primary_{uuid.uuid4().hex[:8]}"

    # Create a primary secret with multiple key-value pairs
    primary_secret_value = {
        "api_key_1": "secret_value_1",
        "api_key_2": "secret_value_2",
        "database_url": "postgresql://user:password@localhost:5432/db",
        "nested_secret": json.dumps({"key": "value", "number": 42}),
    }

    try:
        # Write the primary secret
        write_response = await secret_manager.async_write_secret(
            secret_name=primary_secret_name,
            secret_value=json.dumps(primary_secret_value),
            description="LiteLLM Test Primary Secret",
        )

        print("Primary Secret Write Response:", write_response)
        assert write_response is not None
        assert "ARN" in write_response
        assert "Name" in write_response
        assert write_response["Name"] == primary_secret_name

        # Test reading individual secrets from the primary secret
        for key, expected_value in primary_secret_value.items():
            # Read using the primary_secret_name parameter
            value = await secret_manager.async_read_secret(
                secret_name=key, primary_secret_name=primary_secret_name
            )

            print(f"Read {key} from primary secret:", value)
            assert value == expected_value

        # Test reading a non-existent key from the primary secret
        non_existent_key = "non_existent_key"
        value = await secret_manager.async_read_secret(
            secret_name=non_existent_key, primary_secret_name=primary_secret_name
        )
        assert value is None, f"Expected None for non-existent key, got {value}"

    finally:
        # Cleanup: Delete the primary secret
        delete_response = await secret_manager.async_delete_secret(
            secret_name=primary_secret_name
        )
        print("Delete Response:", delete_response)
        assert delete_response is not None

@pytest.mark.asyncio
async def test_write_secret_with_description_and_tags():
    """Test writing a secret with description and tags"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    test_secret_name = f"litellm_test_{uuid.uuid4().hex[:8]}_tags"
    test_secret_value = "test_value_with_tags"

    test_description = "LiteLLM Secret with Description and Tags"
    test_tags = {
        "Environment": "Test",
        "Owner": "IntelligenceLayer",
        "Purpose": "UnitTest",
    }

    try:
        # Write secret with tags and description
        write_response = await secret_manager.async_write_secret(
            secret_name=test_secret_name,
            secret_value=test_secret_value,
            description=test_description,
            tags=test_tags,
        )

        print("Write Response:", write_response)
        assert write_response is not None
        assert "ARN" in write_response
        assert "Name" in write_response
        assert write_response["Name"] == test_secret_name

        # --- Validate the secret metadata via AWS CLI / boto3 ---
        import boto3

        client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION_NAME"))
        describe_resp = client.describe_secret(SecretId=test_secret_name)
        print("Describe Response:", describe_resp)

        # Validate description
        assert describe_resp.get("Description") == test_description

        # Validate tags (as list of dicts in AWS)
        if "Tags" in describe_resp:
            tag_dict = {t["Key"]: t["Value"] for t in describe_resp["Tags"]}
            for k, v in test_tags.items():
                assert tag_dict.get(k) == v, f"Expected tag {k}={v}, got {tag_dict.get(k)}"
        else:
            pytest.fail("No tags found in describe_secret response")

        # --- Validate secret value ---
        read_value = await secret_manager.async_read_secret(secret_name=test_secret_name)
        print("Read Value:", read_value)
        assert read_value == test_secret_value

    finally:
        # Cleanup: Delete the secret
        delete_response = await secret_manager.async_delete_secret(secret_name=test_secret_name)
        print("Delete Response:", delete_response)
        assert delete_response is not None


def test_secret_manager_with_iam_role_settings():
    """
    Test AWS Secret Manager initialization with IAM role settings
    """
    settings = KeyManagementSettings(
        aws_region_name="us-east-1",
        aws_role_name="arn:aws:iam::123456789012:role/TestRole",
        aws_session_name="test-session",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
    )
    
    # Verify settings are stored
    assert secret_manager.aws_role_name == settings.aws_role_name
    assert secret_manager.aws_region_name == settings.aws_region_name
    assert secret_manager.aws_session_name == settings.aws_session_name


def test_secret_manager_with_cross_account_settings():
    """
    Test AWS Secret Manager initialization with cross-account IAM role settings
    """
    settings = KeyManagementSettings(
        aws_region_name="us-west-2",
        aws_role_name="arn:aws:iam::999999999999:role/CrossAccountRole",
        aws_session_name="cross-account-session",
        aws_external_id="unique-external-id",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
        aws_external_id=settings.aws_external_id,
    )
    
    # Verify settings are stored
    assert secret_manager.aws_role_name == settings.aws_role_name
    assert secret_manager.aws_region_name == settings.aws_region_name
    assert secret_manager.aws_external_id == settings.aws_external_id


def test_secret_manager_with_irsa_settings():
    """
    Test AWS Secret Manager initialization with IRSA (EKS) settings
    """
    settings = KeyManagementSettings(
        aws_region_name="us-east-1",
        aws_role_name="arn:aws:iam::123456789012:role/EKSServiceAccountRole",
        aws_session_name="eks-session",
        aws_web_identity_token="os.environ/AWS_WEB_IDENTITY_TOKEN_FILE",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
        aws_web_identity_token=settings.aws_web_identity_token,
    )
    
    # Verify settings are stored
    assert secret_manager.aws_role_name == settings.aws_role_name
    assert secret_manager.aws_web_identity_token == settings.aws_web_identity_token


def test_secret_manager_with_custom_sts_endpoint():
    """
    Test AWS Secret Manager initialization with custom STS endpoint (VPC endpoint)
    """
    settings = KeyManagementSettings(
        aws_region_name="us-east-1",
        aws_role_name="arn:aws:iam::123456789012:role/VPCRole",
        aws_session_name="vpc-session",
        aws_sts_endpoint="https://sts.us-east-1.vpce-0123456789abcdef.amazonaws.com",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
        aws_sts_endpoint=settings.aws_sts_endpoint,
    )
    
    # Verify settings are stored
    assert secret_manager.aws_role_name == settings.aws_role_name
    assert secret_manager.aws_sts_endpoint == settings.aws_sts_endpoint


def test_secret_manager_with_aws_profile():
    """
    Test AWS Secret Manager initialization with AWS profile
    """
    settings = KeyManagementSettings(
        aws_region_name="us-east-1",
        aws_profile_name="litellm-dev",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_profile_name=settings.aws_profile_name,
    )
    
    # Verify settings are stored
    assert secret_manager.aws_profile_name == settings.aws_profile_name


def test_load_aws_secret_manager_with_settings():
    """
    Test loading AWS Secret Manager with key_management_settings
    """
    import litellm
    
    settings = KeyManagementSettings(
        store_virtual_keys=True,
        aws_region_name="us-east-1",
        aws_role_name="arn:aws:iam::123456789012:role/TestRole",
        aws_session_name="test-session",
    )
    
    # Set environment variable for validation to pass
    os.environ["AWS_REGION_NAME"] = "us-east-1"
    
    try:
        AWSSecretsManagerV2.load_aws_secret_manager(
            use_aws_secret_manager=True,
            key_management_settings=settings,
        )
        
        # Verify the client was created
        assert litellm.secret_manager_client is not None
        assert isinstance(litellm.secret_manager_client, AWSSecretsManagerV2)
        
        # Verify settings were passed through
        assert litellm.secret_manager_client.aws_role_name == settings.aws_role_name
        assert litellm.secret_manager_client.aws_region_name == settings.aws_region_name
        assert litellm.secret_manager_client.aws_session_name == settings.aws_session_name
    finally:
        # Cleanup
        litellm.secret_manager_client = None


@pytest.mark.asyncio
async def test_end_to_end_iam_role_secret_write():
    """
    Test writing a secret using IAM role assumption (integration test)
    
    Requires:
    - AWS_REGION_NAME environment variable
    - TEST_IAM_ROLE_ARN environment variable with ARN of a role that can be assumed
    - Proper AWS credentials configured (via instance profile, IAM role, or environment)
    """
    # Skip if TEST_IAM_ROLE_ARN is not set
    test_role_arn = os.getenv("TEST_IAM_ROLE_ARN")
    if not test_role_arn:
        pytest.skip("TEST_IAM_ROLE_ARN environment variable not set")
    
    aws_region = os.getenv("AWS_REGION_NAME", "us-east-1")
    
    settings = KeyManagementSettings(
        store_virtual_keys=True,
        aws_region_name=aws_region,
        aws_role_name=test_role_arn,
        aws_session_name="integration-test-session",
    )
    
    secret_manager = AWSSecretsManagerV2(
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
    )
    
    test_secret_name = f"litellm_test_iam_{uuid.uuid4().hex[:8]}"
    test_secret_value = "test_value_iam_role"
    
    try:
        # Test write operation using IAM role
        response = await secret_manager.async_write_secret(
            secret_name=test_secret_name,
            secret_value=test_secret_value,
        )
        
        print("Write Response with IAM Role:", response)
        assert response is not None
        assert "ARN" in response
        
        # Test read operation using IAM role
        read_value = await secret_manager.async_read_secret(
            secret_name=test_secret_name
        )
        
        print("Read Value with IAM Role:", read_value)
        assert read_value == test_secret_value
        
    finally:
        # Cleanup: Delete the secret
        try:
            delete_response = await secret_manager.async_delete_secret(
                secret_name=test_secret_name
            )
            print("Delete Response:", delete_response)
        except Exception as e:
            print(f"Cleanup failed: {e}")
