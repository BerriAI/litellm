"""
Test key rotation manager functionality
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import (
    GenerateKeyResponse,
    LiteLLM_VerificationToken,
)
from litellm.proxy.common_utils.key_rotation_manager import KeyRotationManager


class TestKeyRotationManager:
    """Test the KeyRotationManager class functionality."""

    @pytest.mark.asyncio
    async def test_should_rotate_key_logic(self):
        """
        Test the core logic for determining when a key should be rotated.
        
        This tests:
        - Keys with null last_rotation_at should rotate immediately
        - Keys with recent rotation should not rotate
        - Keys with old rotation should rotate
        """
        # Setup
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        now = datetime.now(timezone.utc)
        
        # Test Case 1: Never rotated (last_rotation_at = None) - should rotate
        key_never_rotated = LiteLLM_VerificationToken(
            token="test-token-1",
            auto_rotate=True,
            rotation_interval="30s",
            last_rotation_at=None,
            rotation_count=0
        )
        
        assert manager._should_rotate_key(key_never_rotated, now) == True
        
        # Test Case 2: Recently rotated (10s ago, interval 30s) - should NOT rotate
        key_recently_rotated = LiteLLM_VerificationToken(
            token="test-token-2",
            auto_rotate=True,
            rotation_interval="30s",
            last_rotation_at=now - timedelta(seconds=10),
            rotation_count=1
        )
        
        assert manager._should_rotate_key(key_recently_rotated, now) == False
        
        # Test Case 3: Old rotation (60s ago, interval 30s) - should rotate
        key_old_rotation = LiteLLM_VerificationToken(
            token="test-token-3",
            auto_rotate=True,
            rotation_interval="30s",
            last_rotation_at=now - timedelta(seconds=60),
            rotation_count=2
        )
        
        assert manager._should_rotate_key(key_old_rotation, now) == True
        
        # Test Case 4: No rotation interval - should NOT rotate
        key_no_interval = LiteLLM_VerificationToken(
            token="test-token-4",
            auto_rotate=True,
            rotation_interval=None,
            last_rotation_at=None,
            rotation_count=0
        )
        
        assert manager._should_rotate_key(key_no_interval, now) == False

    @pytest.mark.asyncio
    async def test_find_keys_needing_rotation(self):
        """
        Test finding keys that need rotation from database.
        
        This tests:
        - Only keys with auto_rotate=True and rotation_interval are considered
        - Filtering logic works correctly
        - Database query is constructed properly
        """
        # Setup
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        now = datetime.now(timezone.utc)
        
        # Mock database response
        mock_keys = [
            LiteLLM_VerificationToken(
                token="token-1",
                auto_rotate=True,
                rotation_interval="30s",
                last_rotation_at=None,  # Should rotate
                rotation_count=0
            ),
            LiteLLM_VerificationToken(
                token="token-2",
                auto_rotate=True,
                rotation_interval="60s",
                last_rotation_at=now - timedelta(seconds=30),  # Should NOT rotate (30s < 60s)
                rotation_count=1
            ),
            LiteLLM_VerificationToken(
                token="token-3",
                auto_rotate=True,
                rotation_interval="30s",
                last_rotation_at=now - timedelta(seconds=45),  # Should rotate (45s > 30s)
                rotation_count=2
            )
        ]
        
        mock_prisma_client.db.litellm_verificationtoken.find_many.return_value = mock_keys
        
        # Execute
        keys_needing_rotation = await manager._find_keys_needing_rotation()
        
        # Verify database query
        mock_prisma_client.db.litellm_verificationtoken.find_many.assert_called_once_with(
            where={
                "auto_rotate": True,
                "rotation_interval": {"not": None}
            }
        )
        
        # Verify filtering logic
        assert len(keys_needing_rotation) == 2  # token-1 and token-3 should need rotation
        
        tokens_needing_rotation = [key.token for key in keys_needing_rotation]
        assert "token-1" in tokens_needing_rotation  # Never rotated
        assert "token-2" not in tokens_needing_rotation  # Recently rotated
        assert "token-3" in tokens_needing_rotation  # Old rotation

    @pytest.mark.asyncio
    async def test_rotate_key_updates_database(self):
        """
        Test that key rotation properly updates the database with new rotation info.
        
        This tests:
        - Rotation count is incremented
        - last_rotation_at is set to current time
        - New key token is updated (not old one)
        """
        # Setup
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        # Mock key to rotate
        key_to_rotate = LiteLLM_VerificationToken(
            token="old-token",
            auto_rotate=True,
            rotation_interval="30s",
            last_rotation_at=None,
            rotation_count=0
        )
        
        # Mock regenerate_key_fn response
        mock_response = GenerateKeyResponse(
            key="new-api-key",
            token_id="new-token-id",
            user_id="test-user"
        )
        
        # Mock the regenerate function
        from unittest.mock import patch
        with patch('litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn', return_value=mock_response):
            with patch('litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook'):
                # Execute
                await manager._rotate_key(key_to_rotate)
        
        # Verify database update was called with correct data
        mock_prisma_client.db.litellm_verificationtoken.update.assert_called_once()
        
        call_args = mock_prisma_client.db.litellm_verificationtoken.update.call_args
        
        # Check the WHERE clause targets the new token
        assert call_args[1]["where"]["token"] == "new-token-id"
        
        # Check the data being updated
        update_data = call_args[1]["data"]
        assert update_data["rotation_count"] == 1  # Incremented from 0
        assert "last_rotation_at" in update_data
        assert isinstance(update_data["last_rotation_at"], datetime)
