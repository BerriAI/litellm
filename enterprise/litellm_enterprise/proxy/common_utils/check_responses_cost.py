"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
Cost tracking is handled automatically by litellm.aget_responses().
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
    RESPONSES_COST_POLL_RETRY_ATTEMPTS,
    RESPONSES_COST_POLL_RETRY_DELAY_SECONDS,
    STALE_OBJECT_CLEANUP_MAX_BATCHES_PER_POLL_CYCLE,
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
        """Mark up to `batch_size` stale response rows as `stale_expired`."""
        stale_rows: List[Any] = await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "file_purpose": "response",
                "status": {
                    "not_in": [
                        "completed",
                        "complete",
                        "failed",
                        "expired",
                        "cancelled",
                        "stale_expired",
                    ]
                },
                "created_at": {"lt": cutoff},
            },
            take=batch_size,
            order={"created_at": "asc"},
            select={"id": True},
        )
        stale_ids = [row.id for row in stale_rows]
        if not stale_ids:
            return 0

        await self.prisma_client.db.litellm_managedobjecttable.update_many(
            where={
                "id": {"in": stale_ids},
                "status": {
                    "not_in": [
                        "completed",
                        "complete",
                        "failed",
                        "expired",
                        "cancelled",
                        "stale_expired",
                    ]
                },
            },
            data={"status": "stale_expired"},
        )
        return len(stale_ids)

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
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS
        )
        total_marked = 0
        runs = 0

        for _ in range(STALE_OBJECT_CLEANUP_MAX_BATCHES_PER_POLL_CYCLE):
            result = await self._expire_stale_rows(cutoff, STALE_OBJECT_CLEANUP_BATCH_SIZE)
            runs += 1
            total_marked += result
            if result < STALE_OBJECT_CLEANUP_BATCH_SIZE:
                break

        if total_marked > 0:
            verbose_proxy_logger.warning(
                f"CheckResponsesCost: marked {total_marked} stale managed objects "
                f"(older than {MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days) as stale_expired "
                f"across {runs} cleanup run(s)"
            )

    async def _fetch_response_with_retries(
        self, unified_object_id: str, metadata: Dict[str, str]
    ) -> Optional[Any]:
        from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity

        responses_id_security, _, _ = ResponsesIDSecurity()._decrypt_response_id(
            unified_object_id
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, RESPONSES_COST_POLL_RETRY_ATTEMPTS + 1):
            try:
                return await litellm.aget_responses(
                    response_id=responses_id_security,
                    litellm_metadata=metadata,
                )
            except Exception as e:
                last_error = e
                if attempt == RESPONSES_COST_POLL_RETRY_ATTEMPTS:
                    break
                await asyncio.sleep(RESPONSES_COST_POLL_RETRY_DELAY_SECONDS)

        verbose_proxy_logger.info(
            "Skipping job %s after %d failed poll attempt(s): %s",
            unified_object_id,
            RESPONSES_COST_POLL_RETRY_ATTEMPTS,
            last_error,
        )
        return None

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
        status_to_job_ids: Dict[str, List[str]] = {
            "completed": [],
            "failed": [],
            "cancelled": [],
            "expired": [],
        }

        for job in jobs:
            unified_object_id = job.unified_object_id

            try:
                # Get the stored response object to extract model information
                stored_response = job.file_object or {}
                model_name = stored_response.get("model", None)

                # Prepare metadata with model information for cost tracking
                litellm_metadata = {
                    "user_api_key_user_id": job.created_by or "default-user-id",
                }
                
                # Add model information if available
                if model_name:
                    litellm_metadata["model"] = model_name
                    litellm_metadata["model_group"] = model_name  # Use same value for model_group

                response = await self._fetch_response_with_retries(
                    unified_object_id=unified_object_id,
                    metadata=litellm_metadata,
                )
                if response is None:
                    continue

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
                status_to_job_ids["completed"].append(job.id)
            elif response.status in ["failed", "cancelled", "expired"]:
                verbose_proxy_logger.info(
                    f"Response {unified_object_id} has status {response.status}, marking as {response.status}"
                )
                status_to_job_ids[response.status].append(job.id)

        # Mark completed jobs in the database
        for status, job_ids in status_to_job_ids.items():
            if not job_ids:
                continue
            await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": {"in": job_ids}},
                data={"status": status},
            )
            verbose_proxy_logger.info(f"Marked {len(job_ids)} response jobs as {status}")

