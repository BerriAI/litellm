"""
Regression tests for AWS Secrets Manager same-name in-place rotation fix.

When current_secret_name == new_secret_name (e.g. key alias preserved during
rotation), AWS must use PutSecretValue to update in place instead of
create+delete, which would fail with ResourceExistsException.
"""
from unittest.mock import AsyncMock, patch

import pytest

from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2


@pytest.mark.asyncio
async def test_rotate_secret_same_name_uses_put_secret_value():
    """
    When current_secret_name == new_secret_name, async_rotate_secret should
    call PutSecretValue (async_put_secret_value) instead of create+delete.
    """
    secret_name = "litellm/tenant/litellm-metis-key"
    new_value = "sk-new-rotated-key-value"

    with patch.object(
        AWSSecretsManagerV2,
        "async_put_secret_value",
        new_callable=AsyncMock,
        return_value={"ARN": "arn:aws:secretsmanager:us-east-1:123:secret:test"},
    ) as mock_put:
        with patch.object(
            AWSSecretsManagerV2,
            "async_write_secret",
            new_callable=AsyncMock,
        ) as mock_write:
            with patch.object(
                AWSSecretsManagerV2,
                "async_delete_secret",
                new_callable=AsyncMock,
            ) as mock_delete:
                manager = AWSSecretsManagerV2()
                result = await manager.async_rotate_secret(
                    current_secret_name=secret_name,
                    new_secret_name=secret_name,
                    new_secret_value=new_value,
                )

    # PutSecretValue (in-place update) should be called
    mock_put.assert_called_once_with(
        secret_name=secret_name,
        secret_value=new_value,
        optional_params=None,
        timeout=None,
    )
    # Create + delete should NOT be called
    mock_write.assert_not_called()
    mock_delete.assert_not_called()
    assert result["ARN"] == "arn:aws:secretsmanager:us-east-1:123:secret:test"


@pytest.mark.asyncio
async def test_rotate_secret_different_names_uses_create_delete():
    """
    When current_secret_name != new_secret_name, async_rotate_secret should
    use base class logic (create new, delete old).
    """
    current_name = "litellm/old-key-alias"
    new_name = "litellm/virtual-key-new-token-id"
    new_value = "sk-new-key-value"

    with patch.object(
        AWSSecretsManagerV2,
        "async_read_secret",
        new_callable=AsyncMock,
        side_effect=["sk-old-value", new_value],  # read old, then read new
    ):
        with patch.object(
            AWSSecretsManagerV2,
            "async_write_secret",
            new_callable=AsyncMock,
            return_value={"ARN": "arn:new"},
        ) as mock_write:
            with patch.object(
                AWSSecretsManagerV2,
                "async_delete_secret",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_delete:
                with patch.object(
                    AWSSecretsManagerV2,
                    "async_put_secret_value",
                    new_callable=AsyncMock,
                ) as mock_put:
                    manager = AWSSecretsManagerV2()
                    await manager.async_rotate_secret(
                        current_secret_name=current_name,
                        new_secret_name=new_name,
                        new_secret_value=new_value,
                    )

    # PutSecretValue should NOT be called (different names)
    mock_put.assert_not_called()
    # Create + delete should be called
    mock_write.assert_called_once()
    mock_delete.assert_called_once_with(
        secret_name=current_name,
        recovery_window_in_days=7,
        optional_params=None,
        timeout=None,
    )
