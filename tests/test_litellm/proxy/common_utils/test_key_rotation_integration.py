"""
Regression test for AWS Secrets Manager Auto-Rotation Bug Fix

This test verifies that KeyRotationManager correctly passes key_alias
when calling regenerate_key_fn, ensuring the secret is rotated at the
correct location in AWS Secrets Manager.

Bug Fixed: Key alias was not passed during auto-rotation, causing
secrets to be created at a new location instead of updating in-place.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import (
    GenerateKeyResponse,
    LiteLLM_VerificationToken,
    RegenerateKeyRequest,
)
from litellm.proxy.common_utils.key_rotation_manager import KeyRotationManager


class TestKeyRotationManagerPassesKeyAlias:
    """
    Regression tests to ensure KeyRotationManager passes key_alias
    to regenerate_key_fn during auto-rotation.
    """

    @pytest.mark.asyncio
    async def test_rotate_key_passes_key_alias_to_regenerate_request(self):
        """
        Verify that _rotate_key includes key_alias in the RegenerateKeyRequest.

        This is the core fix: previously, key_alias was NOT passed, causing
        the secret manager hook to use a generated name instead of the alias.
        """
        # Create a mock key with an alias
        test_alias = "tenant1/my-important-key"
        test_token = "sk-test-token-hash-12345"

        mock_key = MagicMock(spec=LiteLLM_VerificationToken)
        mock_key.token = test_token
        mock_key.key_alias = test_alias
        mock_key.key_name = "sk-...1234"
        mock_key.rotation_interval = "30d"
        mock_key.rotation_count = 0

        # Create mock prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_verificationtoken.update = AsyncMock(
            return_value=mock_key
        )

        # Create mock response
        mock_response = GenerateKeyResponse(
            key="sk-new-key-value",
            token_id="new-token-hash",
            key_alias=test_alias,
        )

        # Capture the RegenerateKeyRequest passed to regenerate_key_fn
        captured_request = None

        async def capture_regenerate_key_fn(
            data, user_api_key_dict, litellm_changed_by
        ):
            nonlocal captured_request
            captured_request = data
            return mock_response

        # Patch regenerate_key_fn to capture the request
        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            side_effect=capture_regenerate_key_fn,
        ):
            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                new_callable=AsyncMock,
            ):
                rotation_manager = KeyRotationManager(mock_prisma)
                await rotation_manager._rotate_key(mock_key)

        # CRITICAL ASSERTION: key_alias must be passed
        assert captured_request is not None, "regenerate_key_fn should have been called"
        assert isinstance(captured_request, RegenerateKeyRequest)
        assert captured_request.key == test_token, "Token should be passed correctly"
        assert captured_request.key_alias == test_alias, (
            f"key_alias should be '{test_alias}' but was '{captured_request.key_alias}'. "
            "This is the bug we fixed - key_alias was not being passed!"
        )

    @pytest.mark.asyncio
    async def test_rotate_key_passes_none_alias_when_key_has_no_alias(self):
        """
        Verify that _rotate_key handles keys without an alias gracefully.
        """
        test_token = "sk-test-token-hash-67890"

        mock_key = MagicMock(spec=LiteLLM_VerificationToken)
        mock_key.token = test_token
        mock_key.key_alias = None  # No alias set
        mock_key.key_name = "sk-...5678"
        mock_key.rotation_interval = "30d"
        mock_key.rotation_count = 0

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_verificationtoken.update = AsyncMock(
            return_value=mock_key
        )

        mock_response = GenerateKeyResponse(
            key="sk-new-key-value",
            token_id="new-token-hash",
        )

        captured_request = None

        async def capture_regenerate_key_fn(
            data, user_api_key_dict, litellm_changed_by
        ):
            nonlocal captured_request
            captured_request = data
            return mock_response

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            side_effect=capture_regenerate_key_fn,
        ):
            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                new_callable=AsyncMock,
            ):
                rotation_manager = KeyRotationManager(mock_prisma)
                await rotation_manager._rotate_key(mock_key)

        assert captured_request is not None
        assert captured_request.key == test_token
        assert (
            captured_request.key_alias is None
        ), "key_alias should be None for keys without alias"

class TestKeyRotationSecretNamingStability:
    """
    Tests that the fallback secret name in the rotation hook remains stable
    across rotations to prevent AWS secret sprawl.

    Couple this with the validation fix (Step 1-2) to ensure a stable 
    experience for secret management.
    """

    @pytest.mark.asyncio
    async def test_rotation_hook_uses_initial_secret_name_fallback(self):
        """
        GIVEN: A key WITHOUT an alias (has an initial_secret_name based on token ID)
        WHEN: The key is rotated
        THEN: The hook MUST reuse the existing secret name, NOT generate a new one 
              based on the new token ID.
        """
        from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
        from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth

        # 1. Existing key without alias
        initial_token_hash = "hashed-initial-token"
        existing_key = MagicMock(spec=LiteLLM_VerificationToken)
        existing_key.token = initial_token_hash
        existing_key.key_alias = None
        initial_secret_name = f"virtual-key-{initial_token_hash}"

        # 2. Rotation response (new token ID)
        new_token_id = "hashed-new-token"
        response = GenerateKeyResponse(
            key="sk-new-key",
            token_id=new_token_id,
            key_alias=None
        )

        # 3. Request data without alias
        request_data = RegenerateKeyRequest(
            key=initial_token_hash,
            key_alias=None
        )

        with patch("litellm.proxy.hooks.key_management_event_hooks.KeyManagementEventHooks._rotate_virtual_key_in_secret_manager", new_callable=AsyncMock) as mock_rotate:
            await KeyManagementEventHooks.async_key_rotated_hook(
                data=request_data,
                existing_key_row=existing_key,
                response=response,
                user_api_key_dict=UserAPIKeyAuth(user_role="proxy_admin", api_key="sk-1234", user_id="1234")
            )

            # ASSERT: The new_secret_name MUST be the same as initial_secret_name
            # This ensures PutSecretValue instead of a new secret creation
            mock_rotate.assert_called_once()
            call_kwargs = mock_rotate.call_args.kwargs
            assert call_kwargs["current_secret_name"] == initial_secret_name
            assert call_kwargs["new_secret_name"] == initial_secret_name, \
                f"Secret name drift! Expected {initial_secret_name}, got {call_kwargs['new_secret_name']}. This causes secret sprawl."

    @pytest.mark.asyncio
    async def test_rotation_hook_pre_rotation_alias_consistency(self):
        """
        GIVEN: A key WITH an alias
        WHEN: The key is rotated
        THEN: The hook uses the alias for both current and new names.
        """
        from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
        from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth

        test_alias = "tenant1/stable-key"
        existing_key = MagicMock(spec=LiteLLM_VerificationToken)
        existing_key.token = "old-hash"
        existing_key.key_alias = test_alias

        response = GenerateKeyResponse(token_id="new-hash", key="sk-new", key_alias=test_alias)
        request_data = RegenerateKeyRequest(key="old-hash", key_alias=test_alias)

        with patch("litellm.proxy.hooks.key_management_event_hooks.KeyManagementEventHooks._rotate_virtual_key_in_secret_manager", new_callable=AsyncMock) as mock_rotate:
            await KeyManagementEventHooks.async_key_rotated_hook(
                data=request_data,
                existing_key_row=existing_key,
                response=response,
                user_api_key_dict=UserAPIKeyAuth(user_role="proxy_admin", api_key="sk-123", user_id="1")
            )
            mock_rotate.assert_called_once()
            assert mock_rotate.call_args.kwargs["current_secret_name"] == test_alias
            assert mock_rotate.call_args.kwargs["new_secret_name"] == test_alias

    @pytest.mark.asyncio
    async def test_set_key_rotation_fields_requires_alias(self):
        """
        Tests that _set_key_rotation_fields enforces key_alias requirement
        when secret storage is enabled.
        """
        import litellm
        from litellm.proxy.management_endpoints.key_management_endpoints import _set_key_rotation_fields
        from litellm.proxy._types import ProxyException
        # Create a mock for settings
        mock_settings = MagicMock()
        mock_settings.store_virtual_keys = True

        # Mock settings: store_virtual_keys = True
        with patch("litellm._key_management_settings", mock_settings):
            data = {"auto_rotate": True} # Missing key_alias
            
            # Should raise ProxyException 400
            with pytest.raises(ProxyException) as exc:
                _set_key_rotation_fields(data, auto_rotate=True, rotation_interval="30d")
            
            assert str(exc.value.code) == "400"
            assert "key_alias is required" in str(exc.value.message)

            # Adding key_alias should work
            data["key_alias"] = "valid-alias"
            _set_key_rotation_fields(data, auto_rotate=True, rotation_interval="30d")
            assert data["auto_rotate"] is True
            assert "key_rotation_at" in data
