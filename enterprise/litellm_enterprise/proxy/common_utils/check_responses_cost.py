"""
Polls LiteLLM_ManagedObjectTable to check if the response is complete.
A background response is billed exactly once: whichever client read or poll first sees it
finished moves its row out of a pending status, then logs the body that read already fetched as
one spend event stamped with BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN and the creating key's
attribution, priced against the deployment that served it. The poll's status checks are stamped
with BACKGROUND_RESPONSE_STATUS_POLL_CALL_ORIGIN and client reads are marked managed by the
managed files hook, so both stay free. The spend event carries the id the client holds, the row's
unified_object_id, so its spend log is found under that id with the _background_cost suffix.
A read that finds no row, because the create could not store one, stores it under the reading key
and bills it the same way, so a response the create left untracked is still billed once. A streamed
background create is free on its own; its stream stores the row under the creating key and bills
the finished response through it before the client sees response.completed.
A read the caller retargets with client-side credentials, and a delete of a still-pending response,
settle the bill from a fetch through the configured deployment instead: the client controls what a
retargeted read returns, and a deleted response leaves the poll nothing to fetch.
A row stored before this billing model keeps the encrypted create id as its model_object_id. The
first lookup by that id rekeys it by the provider id, and the poll retires such a row without
billing it once a read has already tracked the response under the provider id.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from litellm_enterprise.proxy.common_utils.managed_object_attribution import (
    ManagedObjectCreatorAttribution,
    ManagedObjectRow,
)
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
    PENDING_RESPONSE_STATUSES,
    STALE_OBJECT_CLEANUP_BATCH_SIZE,
    TERMINAL_RESPONSE_STATUSES,
)
from litellm.proxy.hooks.responses_id_security import decrypted_response_id
from litellm.repositories.table_repositories import ManagedObjectRepository
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import Deployment
from litellm.types.utils import (
    BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
    BACKGROUND_RESPONSE_STATUS_POLL_CALL_ORIGIN,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.repositories.prisma_protocols import TableActions
    from litellm.router import Router

_STORED_RESPONSE_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_RETRIEVE_RESPONSE_CALL_TYPE: Final = "aget_responses"


@dataclass(frozen=True, slots=True)
class _PollTarget:
    response_id: str
    model_metadata: Mapping[str, object]
    deployment: Deployment | None


@dataclass(frozen=True, slots=True)
class BilledBackgroundResponse:
    response: ResponsesAPIResponse
    model: str
    custom_llm_provider: str | None
    litellm_params: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BilledCharge:
    cost: float | None
    logging_obj: "LiteLLMLogging"


BilledResponseLogger = Callable[[BilledBackgroundResponse], Awaitable[BilledCharge]]
UntrackedResponseTracker = Callable[[ResponsesAPIResponse], Awaitable[bool]]


def _without_read_price(response: ResponsesAPIResponse) -> ResponsesAPIResponse:
    unpriced: Final = response.model_copy(deep=True)
    unpriced.hidden_params = {key: value for key, value in response.hidden_params.items() if key != "response_cost"}
    return unpriced


async def log_billed_background_response(billed: BilledBackgroundResponse) -> BilledCharge:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    logging_obj: Final = LiteLLMLogging(
        model=billed.model,
        messages=[{"role": "user", "content": "<retrieve_response>"}],
        stream=False,
        call_type=_RETRIEVE_RESPONSE_CALL_TYPE,
        start_time=datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id=str(uuid.uuid4()),
    )
    logging_obj.update_environment_variables(
        litellm_params={**billed.litellm_params, "metadata": dict(billed.metadata)},
        optional_params={},
        custom_llm_provider=billed.custom_llm_provider,
    )
    unpriced: Final = _without_read_price(billed.response)
    cost: Final = logging_obj.response_cost_calculator(result=unpriced)
    end_time: Final = datetime.now()
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
        async_coroutine=logging_obj.async_success_handler(
            result=unpriced, start_time=logging_obj.start_time, end_time=end_time
        )
    )
    logging_obj.handle_sync_success_callbacks_for_async_calls(
        result=unpriced, start_time=logging_obj.start_time, end_time=end_time
    )
    return BilledCharge(cost=cost, logging_obj=logging_obj)


class _DeploymentBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model: str | None = None


def _deployment_litellm_params(deployment: Deployment) -> Mapping[str, object]:
    from litellm.litellm_core_utils.litellm_logging import mask_api_base_credentials

    api_base: Final = deployment.litellm_params.api_base
    base_model: Final = _DeploymentBaseModel.model_validate(deployment.litellm_params, from_attributes=True).base_model
    return MappingProxyType(
        {
            **({"api_base": mask_api_base_credentials(api_base)} if api_base else {}),
            **({"base_model": base_model} if base_model else {}),
        }
    )


def _deployment_metadata(deployment: Deployment) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "model_info": deployment.model_info.model_dump(exclude_none=True),
            "model_group": deployment.model_name,
            "deployment": deployment.litellm_params.model,
        }
    )


def _managed_object_table(prisma_client: "PrismaClient") -> "TableActions[ManagedObjectRow]":
    return ManagedObjectRepository(prisma_client).table


def _is_response_gone_at_provider(error: Exception, provider_response_id: str) -> bool:
    return getattr(error, "status_code", None) == 404 and provider_response_id in str(error)


def _keyed_by_issued_id(job: ManagedObjectRow) -> bool:
    return decrypted_response_id(job.model_object_id) != job.model_object_id


class CheckResponsesCost:
    def __init__(
        self,
        proxy_logging_obj: "ProxyLogging",
        prisma_client: "PrismaClient",
        llm_router: "Router",
        log_billed_response: BilledResponseLogger = log_billed_background_response,
    ):
        from litellm.proxy.utils import PrismaClient, ProxyLogging
        from litellm.router import Router

        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.llm_router: Router = llm_router
        self._creator_attribution: Final = ManagedObjectCreatorAttribution(prisma_client, "CheckResponsesCost")
        self._log_billed_response: Final = log_billed_response

    def _resolve_deployment(self, response_id: str) -> Deployment | None:
        model_id: str | None = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        return self.llm_router.get_deployment(model_id=model_id) if model_id is not None else None

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

    async def _fetch_unbilled(self, target: _PollTarget) -> ResponsesAPIResponse:
        return await self._get_response(
            response_id=target.response_id,
            litellm_metadata={
                **target.model_metadata,
                INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_STATUS_POLL_CALL_ORIGIN,
            },
            via_router=target.deployment is not None,
        )

    async def _expire_stale_rows(self, cutoff: datetime, batch_size: int) -> int:
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

    def _target(self, response_id: str, model_name: str | None) -> _PollTarget:
        return _PollTarget(
            response_id=response_id,
            model_metadata=MappingProxyType({"model": model_name, "model_group": model_name} if model_name else {}),
            deployment=self._resolve_deployment(response_id),
        )

    def _poll_target(self, job: ManagedObjectRow) -> _PollTarget:
        stored_model: Final = _STORED_RESPONSE_ADAPTER.validate_python(job.file_object).get("model")
        return self._target(
            decrypted_response_id(job.unified_object_id), stored_model if isinstance(stored_model, str) else None
        )

    async def _claim_for_billing(self, job: ManagedObjectRow) -> bool:
        claimed: Final = await _managed_object_table(self.prisma_client).update_many(
            # bounded-ok: PENDING_RESPONSE_STATUSES is a two-value constant tuple
            where={"id": job.id, "status": {"in": list(PENDING_RESPONSE_STATUSES)}},
            data={"status": "completed"},
        )
        return claimed > 0

    async def _release_billing_claim(self, job: ManagedObjectRow) -> None:
        await _managed_object_table(self.prisma_client).update_many(
            where={"id": job.id, "status": "completed"},
            data={"status": job.status or PENDING_RESPONSE_STATUSES[0]},
        )

    async def _billed_response(
        self,
        job: ManagedObjectRow,
        target: _PollTarget,
        response: ResponsesAPIResponse,
    ) -> BilledBackgroundResponse:
        deployment: Final = target.deployment
        return BilledBackgroundResponse(
            response=response.model_copy(update={"id": job.unified_object_id}),
            model=deployment.litellm_params.model if deployment is not None else (response.model or ""),
            custom_llm_provider=ResponsesAPIRequestUtils.decode_responses_api_response_id(target.response_id).get(
                "custom_llm_provider"
            ),
            litellm_params=_deployment_litellm_params(deployment) if deployment is not None else MappingProxyType({}),
            metadata=MappingProxyType(
                {
                    **(await self._creator_attribution.response_metadata(job, job.unified_object_id)),
                    INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
                    **target.model_metadata,
                    **(_deployment_metadata(deployment) if deployment is not None else {}),
                }
            ),
        )

    async def _bill_once(
        self,
        job: ManagedObjectRow,
        target: _PollTarget,
        response: ResponsesAPIResponse,
    ) -> BilledCharge | None:
        if not await self._claim_for_billing(job):
            verbose_proxy_logger.info(f"Response {job.unified_object_id} was already billed elsewhere, skipping")
            return None
        try:
            charge: Final = await self._log_billed_response(await self._billed_response(job, target, response))
        except BaseException:
            await self._release_billing_claim(job)
            raise
        verbose_proxy_logger.info(f"Response {job.unified_object_id} reached a terminal status and was billed")
        return charge

    async def _job_for_provider_id(self, provider_response_id: str) -> ManagedObjectRow | None:
        return await _managed_object_table(self.prisma_client).find_first(
            where={"model_object_id": provider_response_id, "file_purpose": "response"}
        )

    async def _rekey_by_provider_id(self, job: ManagedObjectRow) -> None:
        await _managed_object_table(self.prisma_client).update_many(
            where={"id": job.id}, data={"model_object_id": decrypted_response_id(job.model_object_id)}
        )

    async def _tracked_job(self, response_id: str) -> ManagedObjectRow | None:
        if tracked := await self._job_for_provider_id(decrypted_response_id(response_id)):
            return tracked
        stored_by_issued_id: Final = await _managed_object_table(self.prisma_client).find_first(
            where={"unified_object_id": response_id, "file_purpose": "response"}
        )
        if stored_by_issued_id is not None and _keyed_by_issued_id(stored_by_issued_id):
            await self._rekey_by_provider_id(stored_by_issued_id)
        return stored_by_issued_id

    async def _superseded_by_a_provider_id_row(self, job: ManagedObjectRow) -> bool:
        if not _keyed_by_issued_id(job):
            return False
        if await self._job_for_provider_id(decrypted_response_id(job.model_object_id)) is None:
            await self._rekey_by_provider_id(job)
            return False
        await self._claim_for_billing(job)
        return True

    async def _track_from_deployment(
        self, response_id: str, track_untracked: UntrackedResponseTracker | None
    ) -> BilledCharge | None:
        target: Final = self._target(decrypted_response_id(response_id), None)
        if track_untracked is None or target.deployment is None:
            return None
        return await self.bill_if_finished(response_id, await self._fetch_unbilled(target), track_untracked)

    async def _start_tracking(
        self, response_id: str, response: ResponsesAPIResponse, track_untracked: UntrackedResponseTracker | None
    ) -> ManagedObjectRow | None:
        if track_untracked is None or not await track_untracked(response):
            return None
        return await self._tracked_job(response_id)

    async def bill_if_finished(
        self,
        response_id: str,
        response: ResponsesAPIResponse,
        track_untracked: UntrackedResponseTracker | None = None,
    ) -> BilledCharge | None:
        if response.status not in TERMINAL_RESPONSE_STATUSES:
            return None
        job: Final = await self._tracked_job(response_id) or await self._start_tracking(
            response_id, response, track_untracked
        )
        if job is None or job.status not in PENDING_RESPONSE_STATUSES:
            return None
        return await self._bill_once(job, self._poll_target(job), response)

    async def bill_from_deployment(
        self, response_id: str, track_untracked: UntrackedResponseTracker | None = None
    ) -> BilledCharge | None:
        job: Final = await self._tracked_job(response_id)
        if job is None:
            return await self._track_from_deployment(response_id, track_untracked)
        if job.status not in PENDING_RESPONSE_STATUSES:
            return None
        target: Final = self._poll_target(job)
        response: Final = await self._fetch_unbilled(target)
        if response.status not in TERMINAL_RESPONSE_STATUSES:
            return None
        return await self._bill_once(job, target, response)

    async def check_responses_cost(self):
        """
        Check if background responses are complete and track their cost.
        - Get all status="queued" or "in_progress" and file_purpose="response" jobs
        - Query the provider, unbilled, to check if the response is complete
        - Claim each terminal job by moving its row to completed, then log the body that check fetched
          as its one billed event, so two pollers racing over the same job bill it once
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
                # bounded-ok: PENDING_RESPONSE_STATUSES is a two-value constant tuple
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
                if await self._superseded_by_a_provider_id_row(job):
                    verbose_proxy_logger.info(f"Response {unified_object_id} is billed under its provider id row")
                    continue
                target = self._poll_target(job)
            except Exception as e:
                verbose_proxy_logger.warning(f"Skipping job {unified_object_id} due to error: {e}")
                continue

            provider_response_id = ResponsesAPIRequestUtils.decode_responses_api_response_id(target.response_id).get(
                "response_id", target.response_id
            )
            try:
                response = await self._fetch_unbilled(target)

                verbose_proxy_logger.debug(f"Response {unified_object_id} status: {response.status}")
                if response.status in TERMINAL_RESPONSE_STATUSES:
                    await self._bill_once(job, target, response)

            except Exception as e:
                if target.deployment is not None and _is_response_gone_at_provider(e, provider_response_id):
                    verbose_proxy_logger.info(
                        f"Response {unified_object_id} no longer available at provider (404), marking stale_expired: {e}"
                    )
                    expired_jobs.append(job)
                    continue
                verbose_proxy_logger.warning(f"Skipping job {unified_object_id} due to error: {e}")
                continue

        if len(expired_jobs) > 0:
            await _managed_object_table(self.prisma_client).update_many(
                # bounded-ok: at most MAX_OBJECTS_PER_POLL_CYCLE rows per cycle, the find_many take above
                where={"id": {"in": [job.id for job in expired_jobs]}},
                data={"status": "stale_expired"},
            )
            verbose_proxy_logger.info(f"Marked {len(expired_jobs)} response jobs as stale_expired")
