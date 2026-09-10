"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
Cost tracking is handled by the get-responses call, which prices normally only because the
poll stamps itself with BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN; user-facing reads of the
same route are non-inference and free.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
    PROXY_BATCH_POLLING_INTERVAL,
    STALE_OBJECT_CLEANUP_BATCH_SIZE,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_ManagedObjectTable
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router

TERMINAL_RESPONSE_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})

CLAIM_ABANDONED_AFTER_POLL_CYCLES: Final = 3


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

    async def _get_response(
        self,
        response_id: str,
        litellm_metadata: dict[str, str],
    ) -> ResponsesAPIResponse:
        """Fetch the upstream response through the deployment that served it.

        A LiteLLM-encoded id carries its deployment's ``model_id``, so the router applies that
        deployment's credentials. ``litellm.aget_responses`` only sees provider env vars, so a
        config-only deployment's rows never leave ``queued``.
        """
        model_id: str | None = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        if model_id is None or self.llm_router.get_deployment(model_id=model_id) is None:
            return await litellm.aget_responses(response_id=response_id, litellm_metadata=litellm_metadata)
        router_response = await self.llm_router.aget_responses(
            response_id=response_id, litellm_metadata=litellm_metadata
        )
        return cast(ResponsesAPIResponse, router_response)

    async def _expire_stale_rows(
        self, cutoff: datetime, batch_size: int
    ) -> int:
        """Run the bounded UPDATE that marks stale rows 'stale_expired'.

        PostgreSQL is the only dialect the proxy supports. Same pattern as ``spend_log_cleanup.py``.
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
        """Retire rows stuck in a non-terminal state past the staleness cutoff, so they stop being polled.

        One query with a subquery LIMIT, so a large backlog never lands in Python memory.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS)
        result = await self._expire_stale_rows(cutoff, STALE_OBJECT_CLEANUP_BATCH_SIZE)
        if result > 0:
            verbose_proxy_logger.warning(
                f"CheckResponsesCost: marked {result} stale managed objects "
                f"(older than {MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days) as stale_expired"
            )

    @staticmethod
    def _is_missing_batch_processed_column_error(err: Exception) -> bool:
        message: Final = str(err).lower()
        return "batch_processed" in message or "unknown column" in message or "does not exist" in message

    async def _claim_job_for_costing(self, job: "LiteLLM_ManagedObjectTable") -> bool:
        """Atomically flip batch_processed false to true, returning whether this pod won the row.

        Every pod polls the same table and the read is what prices the job, so the claim has to be
        taken before it. The ``updated_at`` arm takes a claim back from a pod that died holding it;
        ``updated_at`` is ``@updatedAt``, so a live claim is never stolen. A schema without the
        column cannot claim, so it keeps the pre-existing behavior instead of billing nothing.
        """
        abandoned_before: Final = datetime.now(timezone.utc) - timedelta(
            seconds=CLAIM_ABANDONED_AFTER_POLL_CYCLES * PROXY_BATCH_POLLING_INTERVAL
        )
        try:
            claimed: Final = await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={
                    "id": job.id,
                    "OR": [
                        {"batch_processed": False},
                        {"updated_at": {"lt": abandoned_before}},
                    ],
                },
                data={"batch_processed": True},
            )
        except Exception as db_err:
            if self._is_missing_batch_processed_column_error(db_err):
                verbose_proxy_logger.warning(
                    "CheckResponsesCost: batch_processed column not found, billing without a claim"
                )
                return True
            verbose_proxy_logger.error(f"CheckResponsesCost: failed to claim job {job.id} for cost tracking: {db_err}")
            return False
        return claimed > 0

    async def _release_job_claim(self, job: "LiteLLM_ManagedObjectTable") -> None:
        """Give a claimed row back when the read did not bill it, so a later cycle retries it.

        Holding the claim would retire the row unbilled, which is the failure #37050 hit on batches.
        """
        try:
            await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": job.id, "batch_processed": True},
                data={"batch_processed": False},
            )
        except Exception as db_err:
            verbose_proxy_logger.error(
                f"CheckResponsesCost: failed to release the claim on job {job.id}, "
                f"so its cost will not be retried: {db_err}"
            )

    async def _mark_job_completed(self, job: "LiteLLM_ManagedObjectTable") -> None:
        """Retire a billed row from polling, per job so one failure can't strand the rest.

        Only ``status`` is written, and always the literal "completed", because the usage already
        landed in ``LiteLLM_SpendLogs`` and stale-row expiry keys off that exact value.
        """
        try:
            await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": job.id},
                data={"status": "completed"},
            )
        except Exception as db_err:
            verbose_proxy_logger.error(
                f"CheckResponsesCost: failed to mark job {job.id} completed: {db_err}"
            )

    async def check_responses_cost(self):
        """Read every queued background response and retire the ones the provider has finished.

        The read itself is what bills, because it is stamped with the poll's call origin.
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
                
                # Decrypts rows written before model_object_id held the provider's own id.
                responses_id_security, _, _ = ResponsesIDSecurity()._decrypt_response_id(job.model_object_id)
                
                # Prepare metadata with model information for cost tracking
                litellm_metadata = {
                    "user_api_key_user_id": job.created_by or "default-user-id",
                    INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
                    **({"user_api_key_team_id": job.team_id} if job.team_id else {}),
                    **({"user_api_key": job.api_key, "user_api_key_hash": job.api_key} if job.api_key else {}),
                }
                
                # Add model information if available
                if model_name:
                    litellm_metadata["model"] = model_name
                    litellm_metadata["model_group"] = model_name  # Use same value for model_group
                
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} due to error: {e}"
                )
                continue

            if not await self._claim_job_for_costing(job):
                verbose_proxy_logger.debug(
                    f"Response {unified_object_id} is already claimed for costing, leaving it to the claim holder"
                )
                continue

            try:
                response = await self._get_response(
                    response_id=responses_id_security,
                    litellm_metadata=litellm_metadata,
                )
            except Exception as e:
                await self._release_job_claim(job)
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} due to error: {e}"
                )
                continue

            verbose_proxy_logger.debug(
                f"Response {unified_object_id} status: {response.status}, model: {model_name}"
            )

            if response.status not in TERMINAL_RESPONSE_STATUSES:
                await self._release_job_claim(job)
                continue

            verbose_proxy_logger.info(
                f"Response {unified_object_id} has terminal status {response.status}, marking as complete"
            )
            completed_jobs.append(job)

        for job in completed_jobs:
            await self._mark_job_completed(job)

        if len(completed_jobs) > 0:
            verbose_proxy_logger.info(
                f"Marked {len(completed_jobs)} response jobs as completed"
            )
