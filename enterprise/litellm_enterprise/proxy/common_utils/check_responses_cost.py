"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
A background response is billed exactly once: whichever client read or poll first sees it
finished moves its row out of a pending status, then re-reads it stamped with
BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN and the creating key's attribution, the only read
that prices normally; every other read of the route is non-inference and free.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from pydantic import TypeAdapter

from litellm_enterprise.proxy.common_utils.managed_object_attribution import (
    ManagedObjectCreatorAttribution,
    ManagedObjectRow,
)

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
    STALE_OBJECT_CLEANUP_BATCH_SIZE,
)
from litellm.repositories.table_repositories import ManagedObjectRepository
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.repositories.prisma_protocols import TableActions
    from litellm.router import Router

TERMINAL_RESPONSE_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})
PENDING_RESPONSE_STATUSES: Final = ("queued", "in_progress")
_STORED_RESPONSE_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _PollTarget:
    response_id: str
    model_metadata: Mapping[str, object]
    via_router: bool


def _managed_object_table(prisma_client: "PrismaClient") -> "TableActions[ManagedObjectRow]":
    return ManagedObjectRepository(prisma_client).table


def _is_response_gone_at_provider(error: Exception, provider_response_id: str) -> bool:
    return getattr(error, "status_code", None) == 404 and provider_response_id in str(error)


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
        self._creator_attribution: Final = ManagedObjectCreatorAttribution(prisma_client, "CheckResponsesCost")

    def _resolve_deployment(self, response_id: str) -> bool:
        model_id: str | None = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        return model_id is not None and self.llm_router.get_deployment(model_id=model_id) is not None

    async def _get_response(
        self,
        response_id: str,
        litellm_metadata: dict[str, object],
        via_router: bool,
    ) -> ResponsesAPIResponse:
        """Fetch the upstream response, using deployment credentials when available.

        LiteLLM-encoded response IDs carry the ``model_id`` of the deployment that
        served the original request, so routing through ``llm_router`` applies that
        deployment's ``api_base`` / ``api_key`` / ``api_version``, exactly like
        ``GET /v1/responses/{id}`` does. ``litellm.aget_responses`` on its own only
        sees provider env vars, so it fails for every deployment whose credentials
        live in the config; the row then never leaves ``queued``.
        """
        if not via_router:
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

    def _poll_target(self, job: ManagedObjectRow) -> _PollTarget:
        from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity

        stored_model: Final = _STORED_RESPONSE_ADAPTER.validate_python(job.file_object).get("model")
        model_name: Final = stored_model if isinstance(stored_model, str) else None
        responses_id_security, _, _ = ResponsesIDSecurity()._decrypt_response_id(job.unified_object_id)
        return _PollTarget(
            response_id=responses_id_security,
            model_metadata=MappingProxyType({"model": model_name, "model_group": model_name} if model_name else {}),
            via_router=self._resolve_deployment(responses_id_security),
        )

    async def _claim_for_billing(self, job: ManagedObjectRow) -> bool:
        claimed: Final = await _managed_object_table(self.prisma_client).update_many(
            where={"id": job.id, "status": {"in": list(PENDING_RESPONSE_STATUSES)}},
            data={"status": "completed"},
        )
        return claimed > 0

    async def _release_billing_claim(self, job: ManagedObjectRow) -> None:
        await _managed_object_table(self.prisma_client).update_many(
            where={"id": job.id, "status": "completed"},
            data={"status": job.status or PENDING_RESPONSE_STATUSES[0]},
        )

    async def _bill_once(self, job: ManagedObjectRow, target: _PollTarget) -> None:
        if not await self._claim_for_billing(job):
            verbose_proxy_logger.info(f"Response {job.unified_object_id} was already billed elsewhere, skipping")
            return
        billed_read_metadata: Final[dict[str, object]] = {
            **(await self._creator_attribution.build(job, job.unified_object_id)),
            INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
            **target.model_metadata,
        }
        try:
            await self._get_response(target.response_id, billed_read_metadata, target.via_router)
        except Exception:
            await self._release_billing_claim(job)
            raise
        verbose_proxy_logger.info(f"Response {job.unified_object_id} reached a terminal status and was billed")

    async def bill_if_finished(self, unified_object_id: str, status: str) -> None:
        if status not in TERMINAL_RESPONSE_STATUSES:
            return
        job: Final = await _managed_object_table(self.prisma_client).find_first(
            where={
                "unified_object_id": unified_object_id,
                "file_purpose": "response",
                "status": {"in": list(PENDING_RESPONSE_STATUSES)},
            },
        )
        if job is None:
            return
        await self._bill_once(job, self._poll_target(job))

    async def check_responses_cost(self):
        """
        Check if background responses are complete and track their cost.
        - Get all status="queued" or "in_progress" and file_purpose="response" jobs
        - Query the provider, unbilled, to check if the response is complete
        - Claim each terminal job by moving its row to completed, then bill it with one stamped read,
          so two pollers racing over the same job bill it once
        - Mark responses the provider no longer has (404 through a resolved
          deployment) as stale_expired
        """
        try:
            await self._cleanup_stale_managed_objects()
        except Exception as cleanup_err:
            verbose_proxy_logger.warning(
                f"CheckResponsesCost: stale cleanup failed (poll will continue): {cleanup_err}"
            )

        jobs = await _managed_object_table(self.prisma_client).find_many(
            where={
                "status": {"in": list(PENDING_RESPONSE_STATUSES)},
                "file_purpose": "response",
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )
        
        verbose_proxy_logger.debug(f"Found {len(jobs)} response jobs to check")
        expired_jobs: Final[list[ManagedObjectRow]] = []

        for job in jobs:
            unified_object_id = job.unified_object_id

            try:
                target = self._poll_target(job)
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} due to error: {e}"
                )
                continue

            provider_response_id = ResponsesAPIRequestUtils.decode_responses_api_response_id(
                target.response_id
            ).get("response_id", target.response_id)
            try:
                response = await self._get_response(
                    response_id=target.response_id,
                    litellm_metadata=dict(target.model_metadata),
                    via_router=target.via_router,
                )

                verbose_proxy_logger.debug(
                    f"Response {unified_object_id} status: {response.status}"
                )
                if response.status in TERMINAL_RESPONSE_STATUSES:
                    await self._bill_once(job, target)

            except Exception as e:
                if target.via_router and _is_response_gone_at_provider(e, provider_response_id):
                    verbose_proxy_logger.info(
                        f"Response {unified_object_id} no longer available at provider (404), marking stale_expired: {e}"
                    )
                    expired_jobs.append(job)
                    continue
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} due to error: {e}"
                )
                continue

        if len(expired_jobs) > 0:
            await _managed_object_table(self.prisma_client).update_many(
                # bounded-ok: at most MAX_OBJECTS_PER_POLL_CYCLE rows per cycle, the find_many take above
                where={"id": {"in": [job.id for job in expired_jobs]}},
                data={"status": "stale_expired"},
            )
            verbose_proxy_logger.info(
                f"Marked {len(expired_jobs)} response jobs as stale_expired"
            )
