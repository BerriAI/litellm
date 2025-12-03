"""
Tests for BatchIDSecurity hook.

Tests that User B cannot access User A's batches.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.hooks.batch_id_security import BatchIDSecurity
from litellm.types.utils import SpecialEnums


@pytest.mark.asyncio
async def test_user_b_cannot_access_user_a_batch():
    """
    Core security test: User B should get 403 when trying to access User A's batch.
    """
    security = BatchIDSecurity()

    # User B's credentials
    user_b_auth = MagicMock()
    user_b_auth.user_id = "user-b"
    user_b_auth.team_id = "team-b"
    user_b_auth.user_role = None

    # User A's encrypted batch ID
    data = {"batch_id": "batch_encrypted_user_a"}

    with patch.object(security, "_is_encrypted_batch_id", return_value=True):
        with patch.object(
            security,
            "_decrypt_batch_id",
            return_value=("batch_original_123", "user-a", "team-a"),  # User A owns this
        ):
            with patch("litellm.proxy.proxy_server.general_settings", {}):
                with pytest.raises(HTTPException) as exc_info:
                    await security.async_pre_call_hook(
                        user_api_key_dict=user_b_auth,
                        cache=MagicMock(),
                        data=data,
                        call_type="aretrieve_batch",
                    )

                assert exc_info.value.status_code == 403
                assert "Forbidden" in exc_info.value.detail


@pytest.mark.asyncio
async def test_user_a_can_access_own_batch():
    """
    Happy path: User A should successfully access their own batch.
    """
    security = BatchIDSecurity()

    # User A's credentials
    user_a_auth = MagicMock()
    user_a_auth.user_id = "user-a"
    user_a_auth.team_id = "team-a"
    user_a_auth.user_role = None

    # User A's encrypted batch ID
    data = {"batch_id": "batch_encrypted_user_a"}

    with patch.object(security, "_is_encrypted_batch_id", return_value=True):
        with patch.object(
            security,
            "_decrypt_batch_id",
            return_value=("batch_original_123", "user-a", "team-a"),  # User A owns this
        ):
            result = await security.async_pre_call_hook(
                user_api_key_dict=user_a_auth,
                cache=MagicMock(),
                data=data,
                call_type="aretrieve_batch",
            )

            # Should decrypt and return original batch_id
            assert result["batch_id"] == "batch_original_123"
