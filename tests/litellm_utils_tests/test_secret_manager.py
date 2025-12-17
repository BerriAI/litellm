import os
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv
import json

load_dotenv()
import os
import tempfile
from uuid import uuid4

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.llms.azure.azure import get_azure_ad_token_from_oidc
from litellm.llms.bedrock.chat import BedrockConverseLLM, BedrockLLM
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.main import (
    get_secret,
    _should_read_secret_from_secret_manager,
)
from unittest.mock import AsyncMock


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


def test_aws_secret_manager():
    import json

    AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True)

    secret_val = get_secret("litellm_master_key")

    print(f"secret_val: {secret_val}")

    # cast json to dict
    secret_val = json.loads(secret_val)

    assert secret_val["litellm_master_key"] == "sk-1234"


def redact_oidc_signature(secret_val):
    # remove the last part of `.` and replace it with "SIGNATURE_REMOVED"
    return secret_val.split(".")[:-1] + ["SIGNATURE_REMOVED"]


@pytest.mark.skipif(
    os.environ.get("K_SERVICE") is None,
    reason="Cannot run without being in GCP Cloud Run",
)
def test_oidc_google():
    secret_val = get_secret(
        "oidc/google/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")


@pytest.mark.skipif(
    os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN") is None,
    reason="Cannot run without being in GitHub Actions",
)
def test_oidc_github():
    secret_val = get_secret(
        "oidc/github/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circleci():
    secret_val = get_secret("oidc/circleci/")

    print(f"secret_val: {redact_oidc_signature(secret_val)}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circleci_v2():
    secret_val = get_secret(
        "oidc/circleci_v2/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")


@pytest.mark.skip(
    reason="Quarantined: Flaky test - fails with 401 Unauthorized from Azure OAuth. TODO: Switch to our own Azure account or fix authentication"
)
def test_oidc_circleci_with_azure():
    # TODO: Switch to our own Azure account, currently using ai.moda's account
    os.environ["AZURE_TENANT_ID"] = "17c0a27a-1246-4aa1-a3b6-d294e80e783c"
    os.environ["AZURE_CLIENT_ID"] = "4faf5422-b2bd-45e8-a6d7-46543a38acd0"
    azure_ad_token = get_azure_ad_token_from_oidc(
        azure_ad_token="oidc/circleci/",
        azure_client_id=None,
        azure_tenant_id=None,
    )

    print(f"secret_val: {redact_oidc_signature(azure_ad_token)}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circle_v1_with_amazon():
    # The purpose of this test is to get logs using the older v1 of the CircleCI OIDC token

    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci-v1-assume-only"
    aws_web_identity_token = "oidc/circleci/"

    bllm = BedrockLLM()
    creds = bllm.get_credentials(
        aws_region_name="ca-west-1",
        aws_web_identity_token=aws_web_identity_token,
        aws_role_name=aws_role_name,
        aws_session_name="assume-v1-session",
    )


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circle_v1_with_amazon_fips():
    # The purpose of this test is to validate that we can assume a role in a FIPS region

    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci-v1-assume-only"
    aws_web_identity_token = "oidc/circleci/"

    bllm = BedrockConverseLLM()
    creds = bllm.get_credentials(
        aws_region_name="us-west-1",
        aws_web_identity_token=aws_web_identity_token,
        aws_role_name=aws_role_name,
        aws_session_name="assume-v1-session-fips",
        aws_sts_endpoint="https://sts-fips.us-west-1.amazonaws.com",
    )


def test_oidc_env_variable():
    # Create a unique environment variable name
    env_var_name = "OIDC_TEST_PATH_" + uuid4().hex
    os.environ[env_var_name] = "secret-" + uuid4().hex
    secret_val = get_secret(f"oidc/env/{env_var_name}")

    print(f"secret_val: {redact_oidc_signature(secret_val)}")

    assert secret_val == os.environ[env_var_name]

    # now unset the environment variable
    del os.environ[env_var_name]


def test_oidc_file():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        secret_value = "secret-" + uuid4().hex
        temp_file.write(secret_value)
        temp_file.flush()
        temp_file_path = temp_file.name

        secret_val = get_secret(f"oidc/file/{temp_file_path}")

        print(f"secret_val: {redact_oidc_signature(secret_val)}")

        assert secret_val == secret_value


def test_oidc_env_path():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        secret_value = "secret-" + uuid4().hex
        temp_file.write(secret_value)
        temp_file.flush()
        temp_file_path = temp_file.name

        # Create a unique environment variable name
        env_var_name = "OIDC_TEST_PATH_" + uuid4().hex

        # Set the environment variable to the temporary file path
        os.environ[env_var_name] = temp_file_path

        # Test getting the secret using the environment variable
        secret_val = get_secret(f"oidc/env_path/{env_var_name}")

        print(f"secret_val: {redact_oidc_signature(secret_val)}")

        assert secret_val == secret_value

        del os.environ[env_var_name]


@pytest.mark.flaky(retries=6, delay=1)
def test_google_secret_manager():
    """
    Test that we can get a secret from Google Secret Manager
    """
    os.environ["GOOGLE_SECRET_MANAGER_PROJECT_ID"] = "pathrise-convert-1606954137718"

    from litellm.secret_managers.google_secret_manager import GoogleSecretManager

    load_vertex_ai_credentials()
    secret_manager = GoogleSecretManager()

    secret_val = secret_manager.get_secret_from_google_secret_manager(
        secret_name="OPENAI_API_KEY"
    )
    print("secret_val: {}".format(secret_val))

    assert (
        secret_val == "anything"
    ), "did not get expected secret value. expect 'anything', got '{}'".format(
        secret_val
    )


def test_google_secret_manager_read_in_memory():
    """
    Test that Google Secret manager returs in memory value when it exists
    """
    from litellm.secret_managers.google_secret_manager import GoogleSecretManager

    load_vertex_ai_credentials()
    os.environ["GOOGLE_SECRET_MANAGER_PROJECT_ID"] = "pathrise-convert-1606954137718"
    secret_manager = GoogleSecretManager()
    secret_manager.cache.cache_dict["UNIQUE_KEY"] = None
    secret_manager.cache.cache_dict["UNIQUE_KEY_2"] = "lite-llm"

    secret_val = secret_manager.get_secret_from_google_secret_manager(
        secret_name="UNIQUE_KEY"
    )
    print("secret_val: {}".format(secret_val))
    assert secret_val == None

    secret_val = secret_manager.get_secret_from_google_secret_manager(
        secret_name="UNIQUE_KEY_2"
    )
    print("secret_val: {}".format(secret_val))
    assert secret_val == "lite-llm"


def test_should_read_secret_from_secret_manager():
    """
    Test that _should_read_secret_from_secret_manager returns correct values based on access mode
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    # Test when secret manager client is None
    litellm.secret_manager_client = None
    litellm._key_management_settings = KeyManagementSettings()
    assert _should_read_secret_from_secret_manager() is False

    # Test with secret manager client and read_only access
    litellm.secret_manager_client = "dummy_client"
    litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")
    assert _should_read_secret_from_secret_manager() is True

    # Test with secret manager client and read_and_write access
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_and_write"
    )
    assert _should_read_secret_from_secret_manager() is True

    # Test with secret manager client and write_only access
    litellm._key_management_settings = KeyManagementSettings(access_mode="write_only")
    assert _should_read_secret_from_secret_manager() is False

    # Reset global variables
    litellm.secret_manager_client = None
    litellm._key_management_settings = KeyManagementSettings()


def test_get_secret_with_access_mode():
    """
    Test that get_secret respects access mode settings
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    # Set up test environment
    test_secret_name = "TEST_SECRET_KEY"
    test_secret_value = "test_secret_value"
    os.environ[test_secret_name] = test_secret_value

    # Test with write_only access (should read from os.environ)
    litellm.secret_manager_client = "dummy_client"
    litellm._key_management_settings = KeyManagementSettings(access_mode="write_only")
    assert get_secret(test_secret_name) == test_secret_value

    # Test with no KeyManagementSettings but secret_manager_client set
    litellm.secret_manager_client = "dummy_client"
    litellm._key_management_settings = KeyManagementSettings()
    assert _should_read_secret_from_secret_manager() is True

    # Test with read_only access
    litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")
    assert _should_read_secret_from_secret_manager() is True

    # Test with read_and_write access
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_and_write"
    )
    assert _should_read_secret_from_secret_manager() is True

    # Reset global variables
    litellm.secret_manager_client = None
    litellm._key_management_settings = KeyManagementSettings()
    del os.environ[test_secret_name]

def test_key_management_settings_defaults():
    """
    Test that KeyManagementSettings initializes with correct default values.
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    settings = KeyManagementSettings()

    assert settings.store_virtual_keys is False
    assert settings.prefix_for_stored_virtual_keys == "litellm/"
    assert settings.access_mode == "read_only"
    assert settings.description is None
    assert settings.tags is None
    assert settings.primary_secret_name is None


def test_key_management_settings_custom_values():
    """
    Test that KeyManagementSettings correctly stores custom description and tags.
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    custom_tags = {"Environment": "Dev", "Team": "Intelligence"}
    custom_description = "LiteLLM-managed API key for development"

    settings = KeyManagementSettings(
        store_virtual_keys=True,
        prefix_for_stored_virtual_keys="litellm/custom/",
        access_mode="read_and_write",
        primary_secret_name="primary/litellm/keys",
        description=custom_description,
        tags=custom_tags,
    )

    assert settings.store_virtual_keys is True
    assert settings.prefix_for_stored_virtual_keys == "litellm/custom/"
    assert settings.access_mode == "read_and_write"
    assert settings.primary_secret_name == "primary/litellm/keys"
    assert settings.description == custom_description
    assert settings.tags == custom_tags


@pytest.mark.asyncio
async def test_async_write_secret_receives_description_and_tags(monkeypatch):
    """
    Test that AWSSecretsManagerV2.async_write_secret receives description and tags when KeyManagementSettings is set.
    """
    from litellm import litellm
    from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
    from litellm.types.secret_managers.main import KeyManagementSettings

    # Mock out AWS network calls
    mock_async_write = AsyncMock(return_value={"Name": "litellm/test_secret"})
    monkeypatch.setattr(AWSSecretsManagerV2, "async_write_secret", mock_async_write)

    # Setup settings
    litellm._key_management_settings = KeyManagementSettings(
        store_virtual_keys=True,
        description="LiteLLM Unit Test Secret",
        tags={"Owner": "UnitTest", "Purpose": "Validation"},
    )

    # Instantiate fake client
    litellm.secret_manager_client = AWSSecretsManagerV2()

    # Call the helper method that stores a virtual key
    from litellm.proxy.hooks.key_management_event_hooks import (
        KeyManagementEventHooks,
    )

    await KeyManagementEventHooks._store_virtual_key_in_secret_manager(
        secret_name="test_secret", secret_token="test_value"
    )

    # Verify async_write_secret was called with correct metadata
    mock_async_write.assert_called_once()
    args, kwargs = mock_async_write.call_args

    assert kwargs["secret_name"].endswith("test_secret")
    assert kwargs["secret_value"] == "test_value"
    assert kwargs["description"] == "LiteLLM Unit Test Secret"
    assert kwargs["tags"] == {"Owner": "UnitTest", "Purpose": "Validation"}


def test_key_management_settings_serialization_roundtrip():
    """
    Test that KeyManagementSettings serializes and deserializes consistently (Pydantic behavior).
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    original = KeyManagementSettings(
        store_virtual_keys=True,
        prefix_for_stored_virtual_keys="litellm/dev/",
        access_mode="read_and_write",
        description="Roundtrip test",
        tags={"Env": "QA"},
    )

    as_dict = original.model_dump()
    reloaded = KeyManagementSettings(**as_dict)

    assert reloaded.store_virtual_keys is True
    assert reloaded.prefix_for_stored_virtual_keys == "litellm/dev/"
    assert reloaded.access_mode == "read_and_write"
    assert reloaded.description == "Roundtrip test"
    assert reloaded.tags == {"Env": "QA"}
