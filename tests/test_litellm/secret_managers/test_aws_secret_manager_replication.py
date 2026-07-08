"""
Unit tests for AWSSecretsManagerV2 cross-region replication via ReplicateSecretToRegions.

All tests are mocked — no real AWS credentials required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CREATE_RESPONSE = {
    "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:litellm/test-key",
    "Name": "litellm/test-key",
    "VersionId": "mock-version-id",
}

_REPLICATE_RESPONSE = {
    "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:litellm/test-key",
    "ReplicationStatus": [
        {"Region": "us-west-2", "Status": "InProgress"},
    ],
}


def _mock_http_client(json_response: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = json_response
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    return mock_async_client


# ---------------------------------------------------------------------------
# Tests: async_write_secret + replication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_secret_replicates_when_configured():
    """async_replicate_secret is called after a successful CreateSecret when replica_regions is set."""
    manager = AWSSecretsManagerV2(replica_regions=["us-west-2"])

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b'{"Name":"litellm/test-key"}',
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client(_CREATE_RESPONSE),
        ):
            with patch.object(
                AWSSecretsManagerV2,
                "async_replicate_secret",
                new_callable=AsyncMock,
                return_value=_REPLICATE_RESPONSE,
            ) as mock_replicate:
                result = await manager.async_write_secret(
                    secret_name="litellm/test-key",
                    secret_value="sk-test-value",
                )

    assert result == _CREATE_RESPONSE
    mock_replicate.assert_called_once_with(
        secret_name="litellm/test-key",
        replica_regions=["us-west-2"],
        optional_params=None,
        timeout=None,
    )


@pytest.mark.asyncio
async def test_write_secret_no_replication_when_not_configured():
    """async_replicate_secret is NOT called when replica_regions is None."""
    manager = AWSSecretsManagerV2(replica_regions=None)

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b'{"Name":"litellm/test-key"}',
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client(_CREATE_RESPONSE),
        ):
            with patch.object(
                AWSSecretsManagerV2,
                "async_replicate_secret",
                new_callable=AsyncMock,
            ) as mock_replicate:
                result = await manager.async_write_secret(
                    secret_name="litellm/test-key",
                    secret_value="sk-test-value",
                )

    assert result == _CREATE_RESPONSE
    mock_replicate.assert_not_called()


@pytest.mark.asyncio
async def test_replication_failure_does_not_fail_write():
    """If async_replicate_secret raises, async_write_secret still returns the CreateSecret response."""
    manager = AWSSecretsManagerV2(replica_regions=["us-west-2"])

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b'{"Name":"litellm/test-key"}',
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client(_CREATE_RESPONSE),
        ):
            with patch.object(
                AWSSecretsManagerV2,
                "async_replicate_secret",
                new_callable=AsyncMock,
                side_effect=ValueError("AccessDenied: not authorized"),
            ):
                result = await manager.async_write_secret(
                    secret_name="litellm/test-key",
                    secret_value="sk-test-value",
                )

    assert result == _CREATE_RESPONSE


# ---------------------------------------------------------------------------
# Tests: async_replicate_secret directly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_replicate_secret_empty_regions_returns_empty():
    """async_replicate_secret returns {} immediately for an empty list — no HTTP call."""
    manager = AWSSecretsManagerV2()

    with patch(
        "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client"
    ) as mock_get_client:
        result = await manager.async_replicate_secret(
            secret_name="litellm/test-key",
            replica_regions=[],
        )

    assert result == {}
    mock_get_client.assert_not_called()


@pytest.mark.asyncio
async def test_async_replicate_secret_correct_payload():
    """async_replicate_secret sends the correct AddReplicaRegions payload."""
    manager = AWSSecretsManagerV2()
    captured: dict = {}

    def capture_prepare(action, secret_name, optional_params=None, request_data=None):
        captured.update(request_data or {})
        captured["_action"] = action
        return (
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b"{}",
        )

    with patch.object(
        AWSSecretsManagerV2, "_prepare_request", side_effect=capture_prepare
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client(_REPLICATE_RESPONSE),
        ):
            result = await manager.async_replicate_secret(
                secret_name="litellm/test-key",
                replica_regions=["us-west-2", "eu-west-1"],
            )

    assert result == _REPLICATE_RESPONSE
    assert captured["_action"] == "ReplicateSecretToRegions"
    assert captured["SecretId"] == "litellm/test-key"
    assert captured["AddReplicaRegions"] == [
        {"Region": "us-west-2"},
        {"Region": "eu-west-1"},
    ]


@pytest.mark.asyncio
async def test_replication_fires_on_create(caplog):
    """async_replicate_secret emits an INFO log line mentioning ReplicateSecretToRegions."""
    import logging

    manager = AWSSecretsManagerV2()

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b"{}",
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client(_REPLICATE_RESPONSE),
        ):
            with caplog.at_level(logging.INFO, logger="LiteLLM"):
                await manager.async_replicate_secret(
                    secret_name="litellm/test-key",
                    replica_regions=["us-west-2"],
                )

    assert "ReplicateSecretToRegions" in caplog.text


# ---------------------------------------------------------------------------
# Tests: load_aws_secret_manager forwards replica_regions
# ---------------------------------------------------------------------------


def test_load_aws_secret_manager_passes_replica_regions():
    """load_aws_secret_manager must forward replica_regions from key_management_settings."""
    import litellm

    original = litellm.secret_manager_client
    settings = MagicMock()
    settings.aws_region_name = "us-east-1"
    settings.aws_role_name = None
    settings.aws_session_name = None
    settings.aws_external_id = None
    settings.aws_profile_name = None
    settings.aws_web_identity_token = None
    settings.aws_sts_endpoint = None
    settings.replica_regions = ["us-west-2", "eu-west-1"]

    try:
        AWSSecretsManagerV2.load_aws_secret_manager(
            use_aws_secret_manager=True,
            key_management_settings=settings,
        )

        assert isinstance(litellm.secret_manager_client, AWSSecretsManagerV2)
        assert litellm.secret_manager_client.replica_regions == [
            "us-west-2",
            "eu-west-1",
        ]
    finally:
        litellm.secret_manager_client = original


def _http_status_error(status_code: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://secretsmanager.us-east-1.amazonaws.com")
    response = httpx.Response(status_code=status_code, text=body, request=request)
    return httpx.HTTPStatusError(message=body, request=request, response=response)


def _mock_http_client_raising(exc: Exception) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = exc
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    return mock_async_client


# ---------------------------------------------------------------------------
# Tests: error paths in async_write_secret
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_secret_http_error_raises():
    """async_write_secret raises ValueError when CreateSecret returns a non-2xx HTTP status."""
    manager = AWSSecretsManagerV2()

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b'{"Name":"litellm/test-key"}',
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client_raising(
                _http_status_error(400, "ResourceExistsException")
            ),
        ):
            with pytest.raises(ValueError, match="HTTP error occurred"):
                await manager.async_write_secret(
                    secret_name="litellm/test-key",
                    secret_value="sk-test-value",
                )


@pytest.mark.asyncio
async def test_write_secret_timeout_raises():
    """async_write_secret raises ValueError when the CreateSecret call times out."""
    manager = AWSSecretsManagerV2()

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b'{"Name":"litellm/test-key"}',
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client_raising(
                httpx.ReadTimeout("timed out", request=None)
            ),
        ):
            with pytest.raises(ValueError, match="Timeout error occurred"):
                await manager.async_write_secret(
                    secret_name="litellm/test-key",
                    secret_value="sk-test-value",
                )


# ---------------------------------------------------------------------------
# Tests: error paths in async_replicate_secret
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replicate_secret_http_error_raises():
    """async_replicate_secret raises ValueError when ReplicateSecretToRegions returns a non-2xx status."""
    manager = AWSSecretsManagerV2()

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b"{}",
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client_raising(
                _http_status_error(403, "AccessDeniedException")
            ),
        ):
            with pytest.raises(ValueError, match="HTTP error occurred"):
                await manager.async_replicate_secret(
                    secret_name="litellm/test-key",
                    replica_regions=["us-west-2"],
                )


@pytest.mark.asyncio
async def test_replicate_secret_timeout_raises():
    """async_replicate_secret raises ValueError when the ReplicateSecretToRegions call times out."""
    manager = AWSSecretsManagerV2()

    with patch.object(
        AWSSecretsManagerV2,
        "_prepare_request",
        return_value=(
            "https://secretsmanager.us-east-1.amazonaws.com",
            {"Content-Type": "application/x-amz-json-1.1"},
            b"{}",
        ),
    ):
        with patch(
            "litellm.secret_managers.aws_secret_manager_v2.get_async_httpx_client",
            return_value=_mock_http_client_raising(
                httpx.ReadTimeout("timed out", request=None)
            ),
        ):
            with pytest.raises(ValueError, match="Timeout error occurred"):
                await manager.async_replicate_secret(
                    secret_name="litellm/test-key",
                    replica_regions=["us-west-2"],
                )
