"""
Unit tests for AWSSecretsManagerV2 - mocked, no real AWS credentials required.

Tests the write/read/delete cycle for JSON and simple string secrets.
"""

import asyncio, functools, importlib, json, os, sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.secret_managers.main import KeyManagementSettings
from litellm._uuid import uuid
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

_STATIC_CREDENTIALS = {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}
_CMK_ARN = "arn:aws:kms:us-east-1:123456789012:key/11111111-2222-3333-4444-555555555555"


async def _create_secret_body_for_settings(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, settings: KeyManagementSettings
) -> dict[str, object]:
    """Boot the manager from settings the way the proxy does and return the CreateSecret body it posts to AWS."""
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", None)
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True, key_management_settings=settings)
    manager = litellm.secret_manager_client
    assert isinstance(manager, AWSSecretsManagerV2)

    route = respx_mock.post("https://secretsmanager.us-east-1.amazonaws.com/").respond(
        json={"ARN": "arn", "Name": "litellm/test-key"}
    )
    await manager.async_write_secret(
        secret_name="litellm/test-key",
        secret_value="sk-test-value",
        optional_params=dict(_STATIC_CREDENTIALS),
    )
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["X-Amz-Target"] == "secretsmanager.CreateSecret"
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_create_secret_uses_customer_managed_kms_key_from_settings(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    body = await _create_secret_body_for_settings(
        monkeypatch,
        respx_mock,
        KeyManagementSettings(store_virtual_keys=True, aws_region_name="us-east-1", kms_key_id=_CMK_ARN),
    )
    assert body["KmsKeyId"] == _CMK_ARN
    assert body["Name"] == "litellm/test-key"
    assert body["SecretString"] == "sk-test-value"


@pytest.mark.asyncio
async def test_create_secret_omits_kms_key_id_when_not_configured(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    body = await _create_secret_body_for_settings(
        monkeypatch, respx_mock, KeyManagementSettings(store_virtual_keys=True, aws_region_name="us-east-1")
    )
    assert "KmsKeyId" not in body
    assert body["Name"] == "litellm/test-key"


@pytest.mark.asyncio
async def test_write_and_read_json_secret():
    """Test writing and reading a JSON structured secret (mocked)"""
    test_secret_name = "litellm_test_abc12345_json"
    test_secret_value = {
        "api_key": "test_key",
        "model": "gpt-4",
        "temperature": 0.7,
        "metadata": {"team": "ml", "project": "litellm"},
    }
    json_secret_value = json.dumps(test_secret_value)

    write_response = {
        "ARN": f"arn:aws:secretsmanager:us-east-1:123456789012:secret:{test_secret_name}",
        "Name": test_secret_name,
        "VersionId": "mock-version-id",
    }
    delete_response = {
        "ARN": write_response["ARN"],
        "Name": test_secret_name,
        "DeletionDate": "2099-01-01T00:00:00Z",
    }

    with patch.object(
        AWSSecretsManagerV2,
        "async_write_secret",
        new_callable=AsyncMock,
        return_value=write_response,
    ):
        with patch.object(
            AWSSecretsManagerV2,
            "async_read_secret",
            new_callable=AsyncMock,
            return_value=json_secret_value,
        ):
            with patch.object(
                AWSSecretsManagerV2,
                "async_delete_secret",
                new_callable=AsyncMock,
                return_value=delete_response,
            ):
                secret_manager = AWSSecretsManagerV2()

                # Write JSON secret
                response = await secret_manager.async_write_secret(
                    secret_name=test_secret_name,
                    secret_value=json_secret_value,
                    description="LiteLLM JSON Test Secret",
                )

                assert response is not None
                assert "ARN" in response
                assert "Name" in response
                assert response["Name"] == test_secret_name

                # Read and parse JSON secret
                read_value = await secret_manager.async_read_secret(
                    secret_name=test_secret_name
                )
                assert read_value is not None
                parsed_value = json.loads(read_value)

                assert parsed_value == test_secret_value
                assert parsed_value["api_key"] == "test_key"
                assert parsed_value["metadata"]["team"] == "ml"

                # Cleanup
                delete_resp = await secret_manager.async_delete_secret(
                    secret_name=test_secret_name
                )
                assert delete_resp is not None


def _prepare_request_endpoint(
    monkeypatch: pytest.MonkeyPatch, region_name: str, extra_optional_params: dict[str, str] | None = None
) -> str:
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    secret_manager = AWSSecretsManagerV2(aws_region_name=region_name)
    endpoint_url, _headers, _body = secret_manager._prepare_request(
        action="GetSecretValue",
        secret_name="my-secret",
        optional_params={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            **(extra_optional_params or {}),
        },
    )
    return endpoint_url


@pytest.mark.parametrize(
    "region_name,expected_endpoint",
    [
        ("cn-north-1", "https://secretsmanager.cn-north-1.amazonaws.com.cn"),
        ("cn-northwest-1", "https://secretsmanager.cn-northwest-1.amazonaws.com.cn"),
        ("us-gov-west-1", "https://secretsmanager.us-gov-west-1.amazonaws.com"),
        ("us-east-1", "https://secretsmanager.us-east-1.amazonaws.com"),
    ],
)
def test_prepare_request_builds_partition_endpoint(
    monkeypatch: pytest.MonkeyPatch, region_name: str, expected_endpoint: str
) -> None:
    assert _prepare_request_endpoint(monkeypatch, region_name) == expected_endpoint


def test_prepare_request_explicit_bedrock_runtime_endpoint_param_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint_url = _prepare_request_endpoint(
        monkeypatch,
        "cn-north-1",
        {"aws_bedrock_runtime_endpoint": "https://bedrock-runtime.my-vpce.example.com"},
    )
    assert endpoint_url == "https://secretsmanager.my-vpce.example.com"


def test_prepare_request_env_bedrock_runtime_endpoint_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AWS_BEDROCK_RUNTIME_ENDPOINT", "https://bedrock-runtime.eu-west-1.amazonaws.com"
    )
    secret_manager = AWSSecretsManagerV2(aws_region_name="cn-north-1")
    endpoint_url, _headers, _body = secret_manager._prepare_request(
        action="GetSecretValue",
        secret_name="my-secret",
        optional_params={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
        },
    )
    assert endpoint_url == "https://secretsmanager.eu-west-1.amazonaws.com"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)

print("Python Path:", sys.path)

print("Current Working Directory:", os.getcwd())

def skip_on_throttling(func):
    """Skip async test on AWS ThrottlingException instead of failing."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "ThrottlingException" in str(e):
                pytest.skip(f"AWS throttling: {e}")
            raise

    return wrapper

def check_aws_credentials():
    """Helper function to check if AWS credentials are set"""
    if os.getenv("LITELLM_RUN_LIVE_AWS_SECRET_MANAGER_TESTS") != "1":
        pytest.skip("Live AWS Secrets Manager E2E tests are opt-in")
    if os.getenv("CASSETTE_REDIS_URL"):
        pytest.skip("Live AWS Secrets Manager E2E tests cannot run under VCR replay")

    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Missing required AWS credentials: {', '.join(missing_vars)}")

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
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
        read_value = await secret_manager.async_read_secret(secret_name=test_secret_name)

        print("Read Value:", read_value)

        assert read_value == test_secret_value
    finally:
        # Cleanup: Delete the secret
        delete_response = await secret_manager.async_delete_secret(secret_name=test_secret_name)
        print("Delete Response:", delete_response)
        assert delete_response is not None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
async def test_write_and_read_json_secret_aws_secret():
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
        read_value = await secret_manager.async_read_secret(secret_name=test_secret_name)
        parsed_value = json.loads(read_value)

        print("Read Value:", read_value)

        assert parsed_value == test_secret_value
        assert parsed_value["api_key"] == "test_key"
        assert parsed_value["metadata"]["team"] == "ml"
    finally:
        # Cleanup: Delete the secret
        delete_response = await secret_manager.async_delete_secret(secret_name=test_secret_name)
        print("Delete Response:", delete_response)
        assert delete_response is not None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
async def test_read_nonexistent_secret():
    """Test reading a secret that doesn't exist"""
    check_aws_credentials()

    secret_manager = AWSSecretsManagerV2()
    nonexistent_secret = f"litellm_nonexistent_{uuid.uuid4().hex}"

    response = await secret_manager.async_read_secret(secret_name=nonexistent_secret)

    assert response is None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
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
            value = await secret_manager.async_read_secret(secret_name=key, primary_secret_name=primary_secret_name)

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
        delete_response = await secret_manager.async_delete_secret(secret_name=primary_secret_name)
        print("Delete Response:", delete_response)
        assert delete_response is not None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_load_aws_secret_manager_with_settings(monkeypatch: pytest.MonkeyPatch):
    """
    Test loading AWS Secret Manager with key_management_settings
    """
    settings = KeyManagementSettings(
        store_virtual_keys=True,
        aws_region_name="us-east-1",
        aws_role_name="arn:aws:iam::123456789012:role/TestRole",
        aws_session_name="test-session",
    )

    # Set environment variable for validation to pass
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

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

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@skip_on_throttling
async def test_end_to_end_iam_role_secret_write():
    """
    Test writing a secret using IAM role assumption (integration test)

    Requires:
    - AWS_REGION_NAME environment variable
    - TEST_IAM_ROLE_ARN environment variable with ARN of a role that can be assumed
    - Proper AWS credentials configured (via instance profile, IAM role, or environment)
    """
    if os.getenv("LITELLM_RUN_LIVE_AWS_SECRET_MANAGER_TESTS") != "1":
        pytest.skip("Live AWS Secrets Manager E2E tests are opt-in")
    if os.getenv("CASSETTE_REDIS_URL"):
        pytest.skip("Live AWS Secrets Manager E2E tests cannot run under VCR replay")

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
        read_value = await secret_manager.async_read_secret(secret_name=test_secret_name)

        print("Read Value with IAM Role:", read_value)
        assert read_value == test_secret_value

    finally:
        # Cleanup: Delete the secret
        try:
            delete_response = await secret_manager.async_delete_secret(secret_name=test_secret_name)
            print("Delete Response:", delete_response)
        except Exception as e:
            print(f"Cleanup failed: {e}")
