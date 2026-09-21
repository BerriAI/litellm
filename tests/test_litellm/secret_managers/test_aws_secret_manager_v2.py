"""
Unit tests for AWSSecretsManagerV2 - mocked, no real AWS credentials required.

Tests the write/read/delete cycle for JSON and simple string secrets.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.secret_managers.main import KeyManagementSettings

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
