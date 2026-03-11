"""
Test distributed lock behavior for key rotation manager.

Verifies that PodLockManager is correctly used to prevent concurrent
key rotation across multiple pods in a distributed deployment.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.common_utils.key_rotation_manager import KeyRotationManager


class TestKeyRotationLock:
    """Test distributed lock behavior in KeyRotationManager."""

    @pytest.mark.asyncio
    async def test_process_rotations_acquires_lock(self):
        """
        When PodLockManager is provided and lock is acquired,
        rotation logic should run normally.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()  # Redis is available
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Mock _find_keys_needing_rotation to return empty list (no keys to rotate)
        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was acquired
        mock_pod_lock_manager.acquire_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

        # Verify rotation logic ran (cleanup + find keys called)
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

        # Verify lock was released
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_skips_when_lock_held(self):
        """
        When lock is held by another pod, process_rotations() should
        return early without performing any rotation.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock()
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was attempted
        mock_pod_lock_manager.acquire_lock.assert_called_once()

        # Verify rotation logic was NOT executed
        manager._cleanup_expired_deprecated_keys.assert_not_called()
        manager._find_keys_needing_rotation.assert_not_called()

        # Verify lock was NOT released (since it was never acquired)
        mock_pod_lock_manager.release_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_rotations_releases_lock_on_success(self):
        """
        Lock should be released in the finally block after successful rotation.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Simulate finding and rotating a key successfully
        mock_key = LiteLLM_VerificationToken(
            token="test-token",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="test-key",
        )
        manager._find_keys_needing_rotation = AsyncMock(return_value=[mock_key])
        manager._cleanup_expired_deprecated_keys = AsyncMock()
        manager._rotate_key = AsyncMock()

        await manager.process_rotations()

        # Verify rotation was performed
        manager._rotate_key.assert_called_once_with(mock_key)

        # Verify lock was released after success
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_releases_lock_on_error(self):
        """
        Lock should be released in the finally block even if rotation
        throws an exception.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Simulate an error during cleanup
        manager._cleanup_expired_deprecated_keys = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        await manager.process_rotations()

        # Verify lock was still released despite the error
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_works_without_lock_manager(self):
        """
        When pod_lock_manager=None, rotation should run normally
        without any lock logic (backward compat / single-pod mode).
        """
        mock_prisma_client = AsyncMock()

        # No pod_lock_manager provided (default None)
        manager = KeyRotationManager(mock_prisma_client)

        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify rotation logic ran normally
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_rotations_works_without_redis_cache(self):
        """
        When pod_lock_manager exists but redis_cache is None (no Redis configured),
        rotation should run normally without locking.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = None  # No Redis available

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was NOT attempted (no Redis)
        mock_pod_lock_manager.acquire_lock.assert_not_called()

        # Verify rotation logic still ran
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_rotations_handles_none_lock_result(self):
        """
        When acquire_lock returns None (edge case), it should be treated
        as lock NOT acquired, and rotation should be skipped.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=None)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock()
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify rotation logic was NOT executed (None treated as False via `or False`)
        manager._cleanup_expired_deprecated_keys.assert_not_called()
        manager._find_keys_needing_rotation.assert_not_called()

        # Verify lock was NOT released (lock_acquired is False)
        mock_pod_lock_manager.release_lock.assert_not_called()
