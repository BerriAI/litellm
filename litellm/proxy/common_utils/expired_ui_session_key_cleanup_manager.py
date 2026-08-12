"""
Expired UI session key cleanup manager.

Deletes expired virtual keys created for LiteLLM dashboard sessions.
"""

from datetime import datetime, timezone
from typing import Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
    EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
    LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE,
    LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    UI_SESSION_TOKEN_TEAM_ID,
)
from litellm.proxy._types import KeyRequest, LiteLLM_VerificationToken, UserAPIKeyAuth
from litellm.proxy.common_utils.single_owner_job import (
    JobLease,
    WhenLockUnavailable,
    run_as_single_owner,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_verification_tokens,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)


class ExpiredUISessionKeyCleanupManager:
    """
    Cleans up expired UI session keys.
    """

    def __init__(
        self,
        prisma_client: PrismaClient,
        user_api_key_cache: UserApiKeyCache,
        pod_lock_manager: PodLockManager | None = None,
    ):
        self.prisma_client = prisma_client
        self.user_api_key_cache = user_api_key_cache
        self.pod_lock_manager = pod_lock_manager

    async def cleanup_expired_keys(self) -> int:
        """
        Main entry point for deleting expired UI session keys.
        Runs on one elected pod, holding a renewed lease for the whole cycle.
        """
        deleted: Final = await run_as_single_owner(
            pod_lock_manager=self.pod_lock_manager,
            job_name=EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
            ttl_seconds=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
            # Two pods deleting the same batch make the loser's delete 404 and log
            # an error for work that already succeeded
            when_unavailable=WhenLockUnavailable.SKIP,
            run=self._delete_expired_keys,
        )
        return deleted or 0

    async def _delete_expired_keys(self, _lease: JobLease) -> int:
        try:
            verbose_proxy_logger.info("Starting expired UI session key cleanup...")

            expired_keys: Final = await self._find_expired_ui_session_keys()
            if not expired_keys:
                verbose_proxy_logger.debug("No expired UI session keys found")
                return 0

            tokens: Final = [key.token for key in expired_keys if key.token is not None]
            if not tokens:
                return 0

            system_user: Final = UserAPIKeyAuth.get_litellm_internal_jobs_user_api_key_auth()
            response, keys_being_deleted = await delete_verification_tokens(
                tokens=tokens,
                user_api_key_cache=self.user_api_key_cache,
                user_api_key_dict=system_user,
                litellm_changed_by=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            )
            await KeyManagementEventHooks.async_key_deleted_hook(
                data=KeyRequest(keys=tokens),
                keys_being_deleted=keys_being_deleted,
                response=response or {},
                user_api_key_dict=system_user,
                litellm_changed_by=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            )
            deleted_count: Final = self._get_deleted_token_count(
                tokens=tokens,
                response=response,
            )
            verbose_proxy_logger.info("Deleted %s expired UI session key(s)", deleted_count)
            return deleted_count
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                verbose_proxy_logger.debug(
                    "Expired UI session key cleanup skipped because selected keys were already deleted: %s",
                    e,
                )
                return 0
            verbose_proxy_logger.error("Expired UI session key cleanup failed: %s", e)
            return 0

    @staticmethod
    def _get_deleted_token_count(
        tokens: list[str],
        response: dict[str, Any] | None,
    ) -> int:
        """
        Return the number of tokens actually deleted from the delete helper response.
        """
        if response is None:
            return len(tokens)

        deleted_keys: Final = response.get("deleted_keys")
        if isinstance(deleted_keys, list):
            return len(deleted_keys)
        if isinstance(deleted_keys, int):
            return deleted_keys
        if isinstance(deleted_keys, dict):
            nested_deleted_keys: Final = deleted_keys.get("deleted_keys")
            if isinstance(nested_deleted_keys, list):
                return len(nested_deleted_keys)
            if isinstance(nested_deleted_keys, int):
                return nested_deleted_keys

        failed_tokens: Final = response.get("failed_tokens") or []
        if failed_tokens:
            return max(len(tokens) - len(set(failed_tokens)), 0)

        return len(tokens)

    async def _find_expired_ui_session_keys(self) -> list[LiteLLM_VerificationToken]:
        """
        Find expired LiteLLM dashboard session keys.
        """
        now: Final = datetime.now(timezone.utc)
        return await VerificationTokenRepository(self.prisma_client).table.find_many(
            where={
                "team_id": UI_SESSION_TOKEN_TEAM_ID,
                "expires": {"lt": now},
            },
            take=LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE,
        )
