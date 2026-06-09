"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
Cost tracking is handled automatically by litellm.aget_responses().
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
    STALE_OBJECT_CLEANUP_BATCH_SIZE,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router


class CheckResponsesCost:
    def __init__(
        self,
        proxy_logging_obj: "ProxyLogging",
        prisma_client: "PrismaClient",
        llm_router: "Router",
    ):
        from litellm.proxy.utils import PrismaClient, ProxyLogging
        from litellm.router import Router

        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.llm_router: Router = llm_router

    async def _expire_stale_rows(
        self, cutoff: datetime, batch_size: int
    ) -> int:
        """Execute the bounded UPDATE that marks stale rows as 'stale_expired'.

        Isolated so it can be swapped / mocked in tests without touching the
        orchestration logic in ``_cleanup_stale_managed_objects``.

        Uses PostgreSQL syntax (``$1::timestamptz``, ``LIMIT``, double-quoted
        identifiers) which is the only dialect the proxy supports — every
        ``schema.prisma`` in the repo sets ``provider = "postgresql"``.
        Same pattern as ``spend_log_cleanup.py``.
        """
        return await self.prisma_client.db.execute_raw(
            """
            UPDATE "LiteLLM_ManagedObjectTable"
            SET "status" = 'stale_expired'
            WHERE "id" IN (
                SELECT "id" FROM "LiteLLM_ManagedObjectTable"
                WHERE "file_purpose" = 'response'
                AND "status" NOT IN ('completed', 'complete', 'failed', 'expired', 'cancelled', 'stale_expired')
                AND "created_at" < $1::timestamptz
                ORDER BY "created_at" ASC
                LIMIT $2
            )
            """,
            cutoff,
            batch_size,
        )

    async def _cleanup_stale_managed_objects(self) -> None:
        """
        Mark managed objects older than MANAGED_OBJECT_STALENESS_CUTOFF_DAYS days
        in non-terminal states as 'stale_expired'. These will never complete and
        should not be polled.

        Runs as a single DB query with a subquery LIMIT so no rows are loaded
        into Python memory. Processes at most STALE_OBJECT_CLEANUP_BATCH_SIZE
        rows per invocation to avoid overwhelming the DB when there is a large
        backlog.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS)
        result = await self._expire_stale_rows(cutoff, STALE_OBJECT_CLEANUP_BATCH_SIZE)
        if result > 0:
            verbose_proxy_logger.warning(
                f"CheckResponsesCost: marked {result} stale managed objects "
                f"(older than {MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days) as stale_expired"
            )

    async def check_responses_cost(self):
        """
        Check if background responses are complete and track their cost.
        - Get all status="queued" or "in_progress" and file_purpose="response" jobs
        - Query the provider to check if response is complete
        - Cost is automatically tracked by litellm.aget_responses()
        - Mark completed/failed/cancelled responses as complete in the database
        """
        try:
            await self._cleanup_stale_managed_objects()
        except Exception as cleanup_err:
            verbose_proxy_logger.warning(
                f"CheckResponsesCost: stale cleanup failed (poll will continue): {cleanup_err}"
            )

        jobs = await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "status": {"in": ["queued", "in_progress"]},
                "file_purpose": "response",
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )
        
        verbose_proxy_logger.debug(f"Found {len(jobs)} response jobs to check")
        completed_jobs = []

        for job in jobs:
            unified_object_id = job.unified_object_id

            try:
                from litellm.proxy.hooks.responses_id_security import (
                    ResponsesIDSecurity,
                )

                # Get the stored response object to extract model information
                stored_response = job.file_object
                model_name = stored_response.get("model", None)
                
                # Decrypt the response ID
                responses_id_security, _, _ = ResponsesIDSecurity()._decrypt_response_id(unified_object_id)
                
                # Prepare metadata with model information for cost tracking
                litellm_metadata = {
                    "user_api_key_user_id": job.created_by or "default-user-id",
                }
                
                # Add model information if available
                if model_name:
                    litellm_metadata["model"] = model_name
                    litellm_metadata["model_group"] = model_name  # Use same value for model_group
                
                response = await litellm.aget_responses(
                    response_id=responses_id_security,
                    litellm_metadata=litellm_metadata,
                )
                
                verbose_proxy_logger.debug(
                    f"Response {unified_object_id} status: {response.status}, model: {model_name}"
                )
                
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} due to error: {e}"
                )
                continue

            # Check if response is in a terminal state
            if response.status == "completed":
                verbose_proxy_logger.info(
                    f"Response {unified_object_id} is complete. Cost automatically tracked by aget_responses."
                )
                completed_jobs.append(job)
                
            elif response.status in ["failed", "cancelled"]:
                verbose_proxy_logger.info(
                    f"Response {unified_object_id} has status {response.status}, marking as complete"
                )
                completed_jobs.append(job)

        # Mark completed jobs in the database
        if len(completed_jobs) > 0:
            await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": {"in": [job.id for job in completed_jobs]}},
                data={"status": "completed"},
            )
            verbose_proxy_logger.info(
                f"Marked {len(completed_jobs)} response jobs as completed"
            )

