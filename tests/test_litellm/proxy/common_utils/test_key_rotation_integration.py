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
