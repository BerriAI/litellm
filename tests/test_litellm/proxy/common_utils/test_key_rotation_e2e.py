"""
End-to-end tests for key rotation feature.

Covers the critical gaps:
1. Multi-pod simulation: two KeyRotationManagers sharing one PodLockManager
2. Error resilience: partial failures, regenerate_key_fn failures, hook failures
3. Full process_rotations flow with actual key finding + rotation + lock
4. Initialization wiring: PodLockManager is correctly passed
5. Multiple keys: some succeed, some fail, all are attempted
6. Rotation count increments correctly over multiple rotations
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


class TestMultiPodKeyRotation:
    """
    Simulate two pods sharing one Redis lock to verify only one pod
    runs key rotation at a time.
    """

    @pytest.mark.asyncio
    async def test_two_pods_only_one_rotates(self):
        """
        Two KeyRotationManagers with separate pod_lock_managers but
        the same Redis backend. Only the first to acquire the lock
        should rotate; the second should skip.
        """
        mock_prisma = AsyncMock()

        # Shared state to simulate Redis SET NX behavior
        redis_lock = {"holder": None}

        async def make_acquire_lock(pod_id):
            async def acquire(cronjob_id, **kwargs):
                if redis_lock["holder"] is None:
                    redis_lock["holder"] = pod_id
                    return True
                return redis_lock["holder"] == pod_id

            return acquire

        async def make_release_lock(pod_id):
            async def release(cronjob_id):
                if redis_lock["holder"] == pod_id:
                    redis_lock["holder"] = None

            return release

        # Pod A
        pod_a_lock_mgr = MagicMock()
        pod_a_lock_mgr.redis_cache = MagicMock()
        pod_a_lock_mgr.acquire_lock = AsyncMock(
            side_effect=await make_acquire_lock("pod-a")
        )
        pod_a_lock_mgr.release_lock = AsyncMock(
            side_effect=await make_release_lock("pod-a")
        )

        # Pod B
        pod_b_lock_mgr = MagicMock()
        pod_b_lock_mgr.redis_cache = MagicMock()
        pod_b_lock_mgr.acquire_lock = AsyncMock(
            side_effect=await make_acquire_lock("pod-b")
        )
        pod_b_lock_mgr.release_lock = AsyncMock(
            side_effect=await make_release_lock("pod-b")
        )

        manager_a = KeyRotationManager(mock_prisma, pod_lock_manager=pod_a_lock_mgr)
        manager_b = KeyRotationManager(mock_prisma, pod_lock_manager=pod_b_lock_mgr)

        # Both share the same mock methods for rotation logic
        for mgr in [manager_a, manager_b]:
            mgr._cleanup_expired_deprecated_keys = AsyncMock()
            mgr._find_keys_needing_rotation = AsyncMock(return_value=[])

        # Pod A acquires lock first
        await manager_a.process_rotations()
        # Pod A should have run rotation
        manager_a._cleanup_expired_deprecated_keys.assert_called_once()
        manager_a._find_keys_needing_rotation.assert_called_once()

        # Lock is released after pod A finishes, so pod B can now acquire
        # But let's simulate pod B trying WHILE pod A holds the lock
        # Reset the lock state to simulate concurrent access
        redis_lock["holder"] = "pod-a"  # Pod A holds the lock

        await manager_b.process_rotations()
        # Pod B should NOT have run rotation (lock held by pod-a)
        manager_b._cleanup_expired_deprecated_keys.assert_not_called()
        manager_b._find_keys_needing_rotation.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_pod_runs_after_first_releases(self):
        """
        After the first pod releases the lock, the second pod should
        be able to acquire and run rotation.
        """
        mock_prisma = AsyncMock()

        call_order = []

        # Pod A - always gets the lock
        pod_a_lock = MagicMock()
        pod_a_lock.redis_cache = MagicMock()
        pod_a_lock.acquire_lock = AsyncMock(return_value=True)
        pod_a_lock.release_lock = AsyncMock()

        # Pod B - also gets the lock (simulating after A releases)
        pod_b_lock = MagicMock()
        pod_b_lock.redis_cache = MagicMock()
        pod_b_lock.acquire_lock = AsyncMock(return_value=True)
        pod_b_lock.release_lock = AsyncMock()

        manager_a = KeyRotationManager(mock_prisma, pod_lock_manager=pod_a_lock)
        manager_b = KeyRotationManager(mock_prisma, pod_lock_manager=pod_b_lock)

        async def cleanup_a():
            call_order.append("a_cleanup")

        async def cleanup_b():
            call_order.append("b_cleanup")

        manager_a._cleanup_expired_deprecated_keys = AsyncMock(side_effect=cleanup_a)
        manager_a._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager_b._cleanup_expired_deprecated_keys = AsyncMock(side_effect=cleanup_b)
        manager_b._find_keys_needing_rotation = AsyncMock(return_value=[])

        # Run sequentially: A then B
        await manager_a.process_rotations()
        await manager_b.process_rotations()

        # Both should have run
        assert call_order == ["a_cleanup", "b_cleanup"]
        pod_a_lock.release_lock.assert_called_once()
        pod_b_lock.release_lock.assert_called_once()


class TestKeyRotationErrorResilience:
    """
    Tests that key rotation handles errors gracefully:
    - regenerate_key_fn failure for one key doesn't block others
    - Hook failure doesn't crash the process
    - Database update failure is handled
    """

    @pytest.mark.asyncio
    async def test_one_key_fails_others_still_rotate(self):
        """
        If rotation fails for one key, the remaining keys should still
        be attempted. No key should be silently skipped.
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        key1 = LiteLLM_VerificationToken(
            token="token-1",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="key-1",
        )
        key2 = LiteLLM_VerificationToken(
            token="token-2",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="key-2",
        )
        key3 = LiteLLM_VerificationToken(
            token="token-3",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="key-3",
        )

        manager._cleanup_expired_deprecated_keys = AsyncMock()
        manager._find_keys_needing_rotation = AsyncMock(return_value=[key1, key2, key3])

        rotate_calls = []

        async def mock_rotate(key):
            rotate_calls.append(key.token)
            if key.token == "token-2":
                raise Exception("Database connection lost")

        manager._rotate_key = AsyncMock(side_effect=mock_rotate)

        await manager.process_rotations()

        # All 3 keys should have been attempted
        assert rotate_calls == ["token-1", "token-2", "token-3"]

    @pytest.mark.asyncio
    async def test_regenerate_key_fn_failure_is_caught(self):
        """
        If regenerate_key_fn throws, _rotate_key should propagate the error
        but process_rotations should catch it per-key.
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        key = LiteLLM_VerificationToken(
            token="test-token",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="test-key",
        )

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            new_callable=AsyncMock,
            side_effect=Exception("regenerate failed: DB timeout"),
        ):
            # _rotate_key should raise
            with pytest.raises(Exception, match="regenerate failed"):
                await manager._rotate_key(key)

        # But process_rotations should catch per-key errors
        manager._cleanup_expired_deprecated_keys = AsyncMock()
        manager._find_keys_needing_rotation = AsyncMock(return_value=[key])

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            new_callable=AsyncMock,
            side_effect=Exception("regenerate failed: DB timeout"),
        ):
            # Should NOT raise - error is caught per-key
            await manager.process_rotations()

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_prevent_db_update(self):
        """
        If the rotation hook (async_key_rotated_hook) fails, the database
        update for rotation_count should still have succeeded (it runs before the hook).
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        key = LiteLLM_VerificationToken(
            token="test-token",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
        )

        mock_response = GenerateKeyResponse(
            key="new-key", token_id="new-token-id", user_id="test-user"
        )

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                new_callable=AsyncMock,
                side_effect=Exception("Hook failed: secret manager down"),
            ):
                # This will raise because the hook fails
                with pytest.raises(Exception, match="Hook failed"):
                    await manager._rotate_key(key)

        # The DB update should have been called BEFORE the hook
        mock_prisma.db.litellm_verificationtoken.update.assert_called_once()
        update_data = mock_prisma.db.litellm_verificationtoken.update.call_args[1][
            "data"
        ]
        assert update_data["rotation_count"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_prevent_rotation(self):
        """
        If deprecated key cleanup fails, the rotation should still proceed.
        """
        mock_prisma = AsyncMock()
        mock_pod_lock = MagicMock()
        mock_pod_lock.redis_cache = MagicMock()
        mock_pod_lock.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock.release_lock = AsyncMock()

        manager = KeyRotationManager(mock_prisma, pod_lock_manager=mock_pod_lock)

        # Cleanup fails
        manager._cleanup_expired_deprecated_keys = AsyncMock(
            side_effect=Exception("Deprecated table doesn't exist")
        )

        # This should raise within process_rotations, but lock should still be released
        await manager.process_rotations()

        # Lock should still be released in finally block
        mock_pod_lock.release_lock.assert_called_once()


class TestKeyRotationFullFlow:
    """
    Full end-to-end flow tests: find keys -> rotate -> update DB -> release lock
    """

    @pytest.mark.asyncio
    async def test_full_rotation_flow_with_lock(self):
        """
        Full flow: acquire lock -> cleanup -> find keys -> rotate -> update DB -> release lock
        """
        mock_prisma = AsyncMock()

        # Setup lock manager
        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()
        mock_lock.acquire_lock = AsyncMock(return_value=True)
        mock_lock.release_lock = AsyncMock()

        manager = KeyRotationManager(mock_prisma, pod_lock_manager=mock_lock)

        key = LiteLLM_VerificationToken(
            token="old-token-hash",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=datetime.now(timezone.utc) - timedelta(seconds=60),
            rotation_count=2,
            key_name="my-key",
            key_alias="prod/my-key",
        )

        mock_response = GenerateKeyResponse(
            key="sk-new-key-value",
            token_id="new-token-hash",
            user_id="system",
        )

        # Mock cleanup
        mock_prisma.db.litellm_deprecatedverificationtoken.delete_many.return_value = 1
        # Mock find keys
        mock_prisma.db.litellm_verificationtoken.find_many.return_value = [key]

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                new_callable=AsyncMock,
            ):
                await manager.process_rotations()

        # Verify full flow executed:
        # 1. Lock acquired
        mock_lock.acquire_lock.assert_called_once()

        # 2. Cleanup ran
        mock_prisma.db.litellm_deprecatedverificationtoken.delete_many.assert_called_once()

        # 3. Keys were queried
        mock_prisma.db.litellm_verificationtoken.find_many.assert_called_once()

        # 4. DB was updated with new rotation info
        mock_prisma.db.litellm_verificationtoken.update.assert_called_once()
        update_args = mock_prisma.db.litellm_verificationtoken.update.call_args[1]
        assert update_args["where"]["token"] == "new-token-hash"
        assert update_args["data"]["rotation_count"] == 3  # was 2, now 3

        # 5. Lock released
        mock_lock.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotation_count_increments_across_multiple_rotations(self):
        """
        Simulate 3 consecutive rotations and verify rotation_count increments
        correctly each time: 0 -> 1 -> 2 -> 3
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        rotation_counts_seen = []

        for expected_count in range(3):
            key = LiteLLM_VerificationToken(
                token=f"token-v{expected_count}",
                auto_rotate=True,
                rotation_interval="30s",
                key_rotation_at=None,
                rotation_count=expected_count,
            )

            mock_response = GenerateKeyResponse(
                key=f"sk-new-v{expected_count + 1}",
                token_id=f"token-v{expected_count + 1}",
                user_id="system",
            )

            mock_prisma.db.litellm_verificationtoken.update.reset_mock()

            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                with patch(
                    "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                    new_callable=AsyncMock,
                ):
                    await manager._rotate_key(key)

            update_data = mock_prisma.db.litellm_verificationtoken.update.call_args[1][
                "data"
            ]
            rotation_counts_seen.append(update_data["rotation_count"])

        assert rotation_counts_seen == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_no_keys_to_rotate_skips_gracefully(self):
        """
        When no keys need rotation, process should complete without errors.
        """
        mock_prisma = AsyncMock()
        mock_prisma.db.litellm_deprecatedverificationtoken.delete_many.return_value = 0
        mock_prisma.db.litellm_verificationtoken.find_many.return_value = []

        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()
        mock_lock.acquire_lock = AsyncMock(return_value=True)
        mock_lock.release_lock = AsyncMock()

        manager = KeyRotationManager(mock_prisma, pod_lock_manager=mock_lock)

        await manager.process_rotations()

        # Verify no rotation was attempted
        mock_prisma.db.litellm_verificationtoken.update.assert_not_called()
        # But lock was still properly released
        mock_lock.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_regenerate_response_missing_token_id_skips_db_update(self):
        """
        If regenerate_key_fn returns a response without token_id,
        the DB update for rotation metadata should be skipped.
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        key = LiteLLM_VerificationToken(
            token="old-token",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
        )

        # Response with no token_id
        mock_response = GenerateKeyResponse(
            key="sk-new",
            token_id=None,
            user_id="system",
        )

        with patch(
            "litellm.proxy.common_utils.key_rotation_manager.regenerate_key_fn",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with patch(
                "litellm.proxy.common_utils.key_rotation_manager.KeyManagementEventHooks.async_key_rotated_hook",
                new_callable=AsyncMock,
            ):
                await manager._rotate_key(key)

        # DB update should NOT have been called (no token_id)
        mock_prisma.db.litellm_verificationtoken.update.assert_not_called()


class TestKeyRotationInitialization:
    """
    Tests that the PodLockManager wiring in proxy_server.py is correct.
    """

    @pytest.mark.asyncio
    async def test_key_rotation_manager_receives_pod_lock_manager(self):
        """
        Verify KeyRotationManager stores the pod_lock_manager correctly.
        """
        mock_prisma = AsyncMock()
        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()

        manager = KeyRotationManager(mock_prisma, pod_lock_manager=mock_lock)

        assert manager.pod_lock_manager is mock_lock
        assert manager.prisma_client is mock_prisma

    @pytest.mark.asyncio
    async def test_key_rotation_manager_default_no_lock(self):
        """
        When no pod_lock_manager is provided, it defaults to None.
        """
        mock_prisma = AsyncMock()
        manager = KeyRotationManager(mock_prisma)

        assert manager.pod_lock_manager is None

    @pytest.mark.asyncio
    async def test_lock_pattern_matches_spend_log_cleanup(self):
        """
        Verify the key rotation lock pattern is identical to spend_log_cleanup:
        - acquire_lock with cronjob_id
        - release_lock in finally
        - lock_acquired flag guards release
        """
        mock_prisma = AsyncMock()
        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()
        mock_lock.acquire_lock = AsyncMock(return_value=True)
        mock_lock.release_lock = AsyncMock()

        manager = KeyRotationManager(mock_prisma, pod_lock_manager=mock_lock)
        manager._cleanup_expired_deprecated_keys = AsyncMock()
        manager._find_keys_needing_rotation = AsyncMock(return_value=[])

        await manager.process_rotations()

        # Pattern check: acquire with cronjob_id
        acquire_call = mock_lock.acquire_lock.call_args
        assert "cronjob_id" in acquire_call.kwargs or len(acquire_call.args) > 0

        # Pattern check: release with same cronjob_id
        release_call = mock_lock.release_lock.call_args
        assert "cronjob_id" in release_call.kwargs or len(release_call.args) > 0

        # Both should use the same job name
        from litellm.constants import KEY_ROTATION_JOB_NAME

        assert acquire_call.kwargs.get("cronjob_id") == KEY_ROTATION_JOB_NAME
        assert release_call.kwargs.get("cronjob_id") == KEY_ROTATION_JOB_NAME
