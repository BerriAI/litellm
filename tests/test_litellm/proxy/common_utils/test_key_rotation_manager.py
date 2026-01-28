"""
Test key rotation manager functionality
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
        - Keys with null key_rotation_at should rotate immediately
        - Keys with future key_rotation_at should not rotate
        - Keys with past key_rotation_at should rotate
        """
        # Setup
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        now = datetime.now(timezone.utc)
        
        # Test Case 1: No rotation time set (key_rotation_at = None) - should rotate
        key_no_rotation_time = LiteLLM_VerificationToken(
            token="test-token-1",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0
        )
        
        assert manager._should_rotate_key(key_no_rotation_time, now) == True
        
        # Test Case 2: Future rotation time - should NOT rotate
        key_future_rotation = LiteLLM_VerificationToken(
            token="test-token-2",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=now + timedelta(seconds=10),
            rotation_count=1
        )
        
        assert manager._should_rotate_key(key_future_rotation, now) == False
        
        # Test Case 3: Past rotation time - should rotate
        key_past_rotation = LiteLLM_VerificationToken(
            token="test-token-3",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=now - timedelta(seconds=10),
            rotation_count=2
        )
        
        assert manager._should_rotate_key(key_past_rotation, now) == True
        
        # Test Case 4: Exact rotation time - should rotate
        key_exact_rotation = LiteLLM_VerificationToken(
            token="test-token-4",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=now,
            rotation_count=1
        )
        
        assert manager._should_rotate_key(key_exact_rotation, now) == True
        
        # Test Case 5: No rotation interval - should NOT rotate
        key_no_interval = LiteLLM_VerificationToken(
            token="test-token-5",
            auto_rotate=True,
            rotation_interval=None,
            key_rotation_at=None,
            rotation_count=0
        )
        
        assert manager._should_rotate_key(key_no_interval, now) == False

    @pytest.mark.asyncio
    async def test_find_keys_needing_rotation(self):
        """
        Test finding keys that need rotation from database.
        
        This tests:
        - Only keys with auto_rotate=True are considered
        - Database query filters by key_rotation_at properly
        - Keys are returned based on key_rotation_at being null or <= now
        """
        # Setup
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        # Use a fixed timestamp to avoid timing issues in tests
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock database response - these are the keys the database query would return
        mock_keys = [
            LiteLLM_VerificationToken(
                token="token-1",
                auto_rotate=True,
                rotation_interval="30s",
                key_rotation_at=None,  # Should rotate (null key_rotation_at)
                rotation_count=0
            ),
            LiteLLM_VerificationToken(
                token="token-2",
                auto_rotate=True,
                rotation_interval="60s",
                key_rotation_at=now - timedelta(seconds=10),  # Should rotate (past time)
                rotation_count=1
            )
        ]
        
        mock_prisma_client.db.litellm_verificationtoken.find_many.return_value = mock_keys
        
        # Mock datetime.now to return our fixed timestamp
        from unittest.mock import patch
        with patch('litellm.proxy.common_utils.key_rotation_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            # Execute
            keys_needing_rotation = await manager._find_keys_needing_rotation()
        
        # Verify database query - should use OR condition for key_rotation_at
        mock_prisma_client.db.litellm_verificationtoken.find_many.assert_called_once_with(
            where={
                "auto_rotate": True,
                "OR": [
                    {"key_rotation_at": None},
                    {"key_rotation_at": {"lte": now}}
                ]
            }
        )
        
        # Verify all keys returned by database query are included (no additional filtering)
        assert len(keys_needing_rotation) == 2
        
        tokens_needing_rotation = [key.token for key in keys_needing_rotation]
        assert "token-1" in tokens_needing_rotation  # Null key_rotation_at
        assert "token-2" in tokens_needing_rotation  # Past key_rotation_at

    @pytest.mark.asyncio
    async def test_rotate_key_updates_database(self):
        """
        Test that key rotation properly updates the database with new rotation info.
        
        This tests:
        - Rotation count is incremented
        - last_rotation_at is set to current time
        - key_rotation_at is set to next rotation time
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
            key_rotation_at=None,
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
        assert "key_rotation_at" in update_data
        assert isinstance(update_data["key_rotation_at"], datetime)
        
        # Verify key_rotation_at is set to future time (30s from now)
        now = datetime.now(timezone.utc)
        next_rotation = update_data["key_rotation_at"]
        time_diff = (next_rotation - now).total_seconds()
        assert 25 <= time_diff <= 35  # Should be around 30 seconds, allow some tolerance

    @pytest.mark.asyncio
    async def test_rotate_key_sets_grace_period_fields(self):
        """
        Test that key rotation properly sets grace period fields.
        
        This tests:
        - previous_token is set to the old token hash
        - previous_token_expires is set to now + grace_period_minutes
        - grace_period_minutes is preserved
        """
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        key_to_rotate = LiteLLM_VerificationToken(
            token="old-token-hash",
            auto_rotate=True,
            rotation_interval="30s",
            last_rotation_at=None,
            key_rotation_at=None,
            rotation_count=0,
            grace_period_minutes=45,  # Custom grace period
        )
        
        mock_response = GenerateKeyResponse(
            key="new-api-key",
            token_id="new-token-hash",
            user_id="test-user"
        )
        
        with patch('litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn', return_value=mock_response):
            with patch('litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook'):
                await manager._rotate_key(key_to_rotate)
        
        call_args = mock_prisma_client.db.litellm_verificationtoken.update.call_args
        update_data = call_args[1]["data"]
        
        # Check grace period fields
        assert update_data["previous_token"] == "old-token-hash"
        assert "previous_token_expires" in update_data
        assert isinstance(update_data["previous_token_expires"], datetime)
        assert update_data["grace_period_minutes"] == 45
        
        # Verify grace expiry is approximately 45 minutes from now
        now = datetime.now(timezone.utc)
        grace_expiry = update_data["previous_token_expires"]
        time_diff_minutes = (grace_expiry - now).total_seconds() / 60
        assert 44 <= time_diff_minutes <= 46  # Allow 1 minute tolerance

    @pytest.mark.asyncio
    async def test_rotate_key_uses_default_grace_period(self):
        """Test that key rotation uses default 30-minute grace period when not specified."""
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        key_to_rotate = LiteLLM_VerificationToken(
            token="old-token-hash",
            auto_rotate=True,
            rotation_interval="30s",
            rotation_count=0,
            grace_period_minutes=None,  # Not set
        )
        
        mock_response = GenerateKeyResponse(
            key="new-api-key",
            token_id="new-token-hash",
            user_id="test-user"
        )
        
        with patch('litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn', return_value=mock_response):
            with patch('litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook'):
                await manager._rotate_key(key_to_rotate)
        
        call_args = mock_prisma_client.db.litellm_verificationtoken.update.call_args
        update_data = call_args[1]["data"]
        assert update_data["grace_period_minutes"] == 30
        
        now = datetime.now(timezone.utc)
        grace_expiry = update_data["previous_token_expires"]
        time_diff_minutes = (grace_expiry - now).total_seconds() / 60
        assert 29 <= time_diff_minutes <= 31

    @pytest.mark.asyncio
    async def test_cleanup_expired_grace_period_tokens(self):
        """Test that cleanup_expired_grace_period_tokens removes expired previous tokens."""
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        mock_prisma_client.db.litellm_verificationtoken.update_many.return_value = 3
        
        await manager.cleanup_expired_grace_period_tokens()
        
        mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once()
        call_args = mock_prisma_client.db.litellm_verificationtoken.update_many.call_args
        
        where_clause = call_args[1]["where"]
        assert where_clause["previous_token"] == {"not": None}
        assert "previous_token_expires" in where_clause
        assert "lt" in where_clause["previous_token_expires"]
        
        data_clause = call_args[1]["data"]
        assert data_clause["previous_token"] is None
        assert data_clause["previous_token_expires"] is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_grace_period_tokens_handles_exception(self):
        """Test that cleanup handles database exceptions gracefully without raising."""
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        mock_prisma_client.db.litellm_verificationtoken.update_many.side_effect = Exception(
            "Database error"
        )
        
        # Should not raise, just log the error
        await manager.cleanup_expired_grace_period_tokens()
        
        # Verify the method was called (and failed)
        mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_grace_period_tokens_no_expired_tokens(self):
        """Test cleanup when there are no expired tokens to clean up."""
        mock_prisma_client = AsyncMock()
        manager = KeyRotationManager(mock_prisma_client)
        
        # No tokens updated
        mock_prisma_client.db.litellm_verificationtoken.update_many.return_value = 0
        
        await manager.cleanup_expired_grace_period_tokens()
        
        # Should complete without issues
        mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once()
