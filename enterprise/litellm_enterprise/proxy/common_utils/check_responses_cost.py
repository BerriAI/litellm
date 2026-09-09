"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
Cost tracking is handled by the get-responses call, which prices normally only because the
poll stamps itself with BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN; user-facing reads of the
same route are non-inference and free.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Final, Optional, cast

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
        litellm_metadata: Dict[str, str],
    ) -> ResponsesAPIResponse:
        """Fetch the upstream response, using deployment credentials when available.

        LiteLLM-encoded response IDs carry the ``model_id`` of the deployment that
        served the original request, so routing through ``llm_router`` applies that
        deployment's ``api_base`` / ``api_key`` / ``api_version``, exactly like
        ``GET /v1/responses/{id}`` does. ``litellm.aget_responses`` on its own only
        sees provider env vars, so it fails for every deployment whose credentials
        live in the config; the row then never leaves ``queued``.
        """
        model_id: Optional[str] = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        if model_id is None or self.llm_router.get_deployment(model_id=model_id) is None:
            return await litellm.aget_responses(response_id=response_id, litellm_metadata=litellm_metadata)
        router_response = await self.llm_router.aget_responses(
            response_id=response_id, litellm_metadata=litellm_metadata
        )
        return cast(ResponsesAPIResponse, router_response)

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

    @staticmethod
    def _is_missing_batch_processed_column_error(err: Exception) -> bool:
        message: Final = str(err).lower()
        return "batch_processed" in message or "unknown column" in message or "does not exist" in message

    async def _claim_job_for_costing(self, job: "LiteLLM_ManagedObjectTable") -> bool:
        """Atomically flip batch_processed from false to true, returning whether this pod won the row.

        Every pod and uvicorn worker schedules its own CheckResponsesCost against the shared table,
        so without this compare-and-swap two of them select the same queued response in one window
        and both bill it. The claim is taken before the read because the read is what prices the
        job: ``aget_responses`` stamped with the poll origin writes the spend log itself, so there
        is no later point at which to serialize. Schemas without the column can't be claimed, so
        they keep the pre-existing behavior rather than silently billing nothing.

        A pod that dies between winning the claim and billing would otherwise strand the row:
        it holds a claim nobody will release, and its status never reaches terminal, so every
        later cycle re-selects it and loses. The ``updated_at`` arm takes such a claim back once
        it has gone unbilled for longer than any live cycle could hold it. ``updated_at`` is
        ``@updatedAt``, so a healthy in-flight claim refreshed moments ago is never stolen.
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
        """Give a claimed row back when the read did not bill it, so a later poll cycle retries it.

        A response still queued at the provider, or whose read raised, has no spend to record yet.
        Holding the claim would retire it permanently, which is the failure #37050 hit on batches.
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

        Only ``status`` is written. The generation's usage and spend already land in
        ``LiteLLM_SpendLogs`` unconditionally, so copying the response body onto this row would
        duplicate content the provider still serves, on a table nothing ever deletes from.

        ``status`` stays the literal "completed" for every terminal provider status, matching
        what this poller has always written, so stale-row expiry keeps skipping these rows.
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
        """
        Check if background responses are complete and track their cost.
        - Get all status="queued" or "in_progress" and file_purpose="response" jobs
        - Query the provider to check if response is complete
        - Cost is tracked by the get-responses call, billed because the poll is stamped
          with BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN
        - Mark responses in a terminal state as complete in the database
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
