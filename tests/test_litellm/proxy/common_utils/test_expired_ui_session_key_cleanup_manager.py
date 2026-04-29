"""
Test expired UI session key cleanup manager functionality.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.constants import (
    EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
    LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE,
    LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    UI_SESSION_TOKEN_TEAM_ID,
)
from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager import (
    ExpiredUISessionKeyCleanupManager,
)


class TestExpiredUISessionKeyCleanupManager:
    """Test the ExpiredUISessionKeyCleanupManager class functionality."""

    @pytest.mark.asyncio
    async def test_find_expired_ui_session_keys_filters_dashboard_team_and_expiry(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )

        now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
        mock_keys = [
            LiteLLM_VerificationToken(
                token="expired-dashboard-token",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=now - timedelta(seconds=1),
            )
        ]
        mock_prisma_client.db.litellm_verificationtoken.find_many.return_value = (
            mock_keys
        )

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )

            keys = await manager._find_expired_ui_session_keys()

        mock_prisma_client.db.litellm_verificationtoken.find_many.assert_called_once_with(
            where={
                "team_id": UI_SESSION_TOKEN_TEAM_ID,
                "expires": {"lt": now},
            },
            take=LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE,
        )
        assert keys == mock_keys

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_uses_existing_delete_path(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        expired_key = LiteLLM_VerificationToken(
            token="expired-dashboard-token",
            team_id=UI_SESSION_TOKEN_TEAM_ID,
            expires=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        manager._find_expired_ui_session_keys = AsyncMock(return_value=[expired_key])

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            mock_delete_verification_tokens.return_value = (
                {"deleted_keys": ["expired-dashboard-token"], "failed_tokens": []},
                [expired_key],
            )
            with patch(
                "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.KeyManagementEventHooks.async_key_deleted_hook",
                new_callable=AsyncMock,
            ) as mock_key_deleted_hook:
                deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 1
        mock_delete_verification_tokens.assert_called_once()
        call_kwargs = mock_delete_verification_tokens.call_args.kwargs
        assert call_kwargs["tokens"] == ["expired-dashboard-token"]
        assert call_kwargs["user_api_key_cache"] == mock_cache
        assert (
            call_kwargs["litellm_changed_by"]
            == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME
        )
        assert call_kwargs["user_api_key_dict"].user_id == "system"
        mock_key_deleted_hook.assert_called_once()
        hook_kwargs = mock_key_deleted_hook.call_args.kwargs
        assert hook_kwargs["data"].keys == ["expired-dashboard-token"]
        assert hook_kwargs["keys_being_deleted"] == [expired_key]
        assert hook_kwargs["response"] == {
            "deleted_keys": ["expired-dashboard-token"],
            "failed_tokens": [],
        }
        assert (
            hook_kwargs["litellm_changed_by"]
            == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME
        )

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_deletes_multiple_keys(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        expired_keys = [
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-1",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-2",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        ]
        tokens = [key.token for key in expired_keys]
        manager._find_expired_ui_session_keys = AsyncMock(return_value=expired_keys)

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            mock_delete_verification_tokens.return_value = (
                {"deleted_keys": tokens, "failed_tokens": []},
                expired_keys,
            )
            with patch(
                "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.KeyManagementEventHooks.async_key_deleted_hook",
                new_callable=AsyncMock,
            ) as mock_key_deleted_hook:
                deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 2
        assert mock_delete_verification_tokens.call_args.kwargs["tokens"] == tokens
        hook_kwargs = mock_key_deleted_hook.call_args.kwargs
        assert hook_kwargs["data"].keys == tokens
        assert hook_kwargs["keys_being_deleted"] == expired_keys
        assert hook_kwargs["response"] == {"deleted_keys": tokens, "failed_tokens": []}

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_returns_successful_delete_count(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        expired_keys = [
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-1",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-2",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        ]
        tokens = [key.token for key in expired_keys]
        manager._find_expired_ui_session_keys = AsyncMock(return_value=expired_keys)

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            mock_delete_verification_tokens.return_value = (
                {
                    "deleted_keys": ["expired-dashboard-token-1"],
                    "failed_tokens": ["expired-dashboard-token-2"],
                },
                [expired_keys[0]],
            )
            with patch(
                "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.KeyManagementEventHooks.async_key_deleted_hook",
                new_callable=AsyncMock,
            ):
                deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 1
        assert mock_delete_verification_tokens.call_args.kwargs["tokens"] == tokens

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_counts_nested_delete_response(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        expired_keys = [
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-1",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
            LiteLLM_VerificationToken(
                token="expired-dashboard-token-2",
                team_id=UI_SESSION_TOKEN_TEAM_ID,
                expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        ]
        tokens = [key.token for key in expired_keys]
        manager._find_expired_ui_session_keys = AsyncMock(return_value=expired_keys)

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            mock_delete_verification_tokens.return_value = (
                {
                    "deleted_keys": {"deleted_keys": 2},
                    "failed_tokens": tokens,
                },
                expired_keys,
            )
            with patch(
                "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.KeyManagementEventHooks.async_key_deleted_hook",
                new_callable=AsyncMock,
            ):
                deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_treats_missing_keys_as_noop(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        expired_key = LiteLLM_VerificationToken(
            token="expired-dashboard-token",
            team_id=UI_SESSION_TOKEN_TEAM_ID,
            expires=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        manager._find_expired_ui_session_keys = AsyncMock(return_value=[expired_key])

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            mock_delete_verification_tokens.side_effect = HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "No keys found"},
            )
            with patch(
                "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.KeyManagementEventHooks.async_key_deleted_hook",
                new_callable=AsyncMock,
            ) as mock_key_deleted_hook:
                deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 0
        mock_key_deleted_hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_noops_when_no_keys_found(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )
        manager._find_expired_ui_session_keys = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager.delete_verification_tokens",
            new_callable=AsyncMock,
        ) as mock_delete_verification_tokens:
            deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 0
        mock_delete_verification_tokens.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_skips_when_lock_held(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            pod_lock_manager=mock_pod_lock_manager,
        )
        manager._find_expired_ui_session_keys = AsyncMock()

        deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 0
        mock_pod_lock_manager.acquire_lock.assert_called_once_with(
            cronjob_id=EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
        )
        manager._find_expired_ui_session_keys.assert_not_called()
        mock_pod_lock_manager.release_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys_releases_acquired_lock(self):
        mock_prisma_client = AsyncMock()
        mock_cache = MagicMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = ExpiredUISessionKeyCleanupManager(
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            pod_lock_manager=mock_pod_lock_manager,
        )
        manager._find_expired_ui_session_keys = AsyncMock(return_value=[])

        deleted_count = await manager.cleanup_expired_keys()

        assert deleted_count == 0
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id=EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
        )
