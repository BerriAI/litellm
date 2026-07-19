"""
Polls LiteLLM_ManagedObjectTable to check if the batch job is complete, and if the cost has been tracked.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    ABANDONED_PRICING_CLAIM_RECLAIM_SECONDS,
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
)

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.proxy._types import LiteLLM_ManagedObjectTable
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
    from litellm.types.utils import LiteLLMBatch


CHECK_BATCH_COST_USER_AGENT = "LiteLLM Proxy/CheckBatchCost"

# Row-local dedup marker set the moment spend side effects have run. The
# SpendLogs request_id lookup is the primary dedup, but it is inert when the
# operator sets disable_spend_logs — the counters (key/user/team/daily spend)
# still increment, so without a marker on the billing row itself a worker
# dying between spend emission and finalization would re-bill them after
# claim reclamation (codex P1 round 5).
SPEND_RECORDED_MARKER_KEY = "batch_cost_spend_recorded"


class CheckBatchCost:
    def __init__(
        self,
        proxy_logging_obj: "ProxyLogging",
        prisma_client: "PrismaClient",
        llm_router: "Router",
        track_unmanaged_batch_cost: bool = False,
    ):
        from litellm.proxy.utils import PrismaClient, ProxyLogging
        from litellm.router import Router

        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.llm_router: Router = llm_router
        self._track_unmanaged_batch_cost = track_unmanaged_batch_cost
        # Cached after the first poll cycle. Once we know the column is absent we skip
        # the guaranteed-failing primary query on every subsequent cycle.
        self._has_batch_processed_column: bool = True

    @staticmethod
    def _get_job_file_object(job: Any) -> Dict[str, Any]:
        """The job row's file_object as a dict. The hook write-path stores it
        as a JSON string, so tolerate one level of string encoding; anything
        unparseable returns {}."""
        file_object = getattr(job, "file_object", None)
        for _ in range(2):
            if not isinstance(file_object, str):
                break
            try:
                file_object = json.loads(file_object)
            except ValueError:
                return {}
        return file_object if isinstance(file_object, dict) else {}

    @classmethod
    def _get_job_attribution(cls, job: Any) -> Dict[str, Any]:
        """Attribution stashed in file_object.litellm_attribution by routes
        that register batches directly (the /v1/messages/batches route) —
        user_api_key (hash), user_api_key_team_id, user_api_key_end_user_id,
        user_api_key_alias. Rows without the stash return {} and keep the
        pre-existing behavior."""
        attribution = cls._get_job_file_object(job).get("litellm_attribution")
        return attribution if isinstance(attribution, dict) else {}

    async def _claim_job(self, job: Any) -> bool:
        """Atomically claim a row before pricing: every proxy process runs
        this poller, and pricing + flag-flip were previously non-atomic, so
        concurrent workers could each bill the same batch (codex P1). The
        conditional update_many means exactly one worker wins the claim.
        Schemas without the batch_processed column cannot claim atomically
        and keep the legacy single-worker assumption."""
        if not self._has_batch_processed_column:
            return True
        try:
            claimed_count = await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": job.id, "batch_processed": False},
                data={"batch_processed": True, "status": "pricing"},
            )
        except Exception as claim_err:
            verbose_proxy_logger.error(
                f"CheckBatchCost: claim failed for job {job.id}; skipping this cycle: {claim_err}"
            )
            return False
        return claimed_count == 1

    async def _release_job_claim(self, job: Any) -> None:
        """Return a claimed-but-unpriced row to the pool so the next poll
        retries it. Fenced to the claim-held state: after a reclaim, a slow
        ex-owner's release must not flip a row another worker has since
        claimed or finalized (codex P2 round 3). Best effort: an orphaned
        claim (release also failed) is recovered by
        _reclaim_abandoned_pricing_claims on later cycles — the deterministic
        litellm_call_id backstops the double-bill side."""
        try:
            if not self._has_batch_processed_column:
                await self.prisma_client.db.litellm_managedobjecttable.update(
                    where={"id": job.id},
                    data={"batch_processed": False, "status": "validating"},
                )
                return
            released = await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": job.id, "status": "pricing", "batch_processed": True},
                data={"batch_processed": False, "status": "validating"},
            )
            if released != 1:
                verbose_proxy_logger.warning(
                    f"CheckBatchCost: release skipped for job {job.id} — the claim was taken over by another worker"
                )
        except Exception as release_err:
            verbose_proxy_logger.error(
                f"CheckBatchCost: failed to release claim on job {job.id}: {release_err}"
            )

    async def _finalize_job(self, job: Any, update_data: Dict[str, Any]) -> bool:
        """Write the terminal row state, fenced to the claim this worker
        holds: a slow ex-owner whose claim was reclaimed must not overwrite
        the new owner's finalization (codex P2 round 3). Legacy schemas
        without batch_processed keep the plain unconditional update."""
        if not self._has_batch_processed_column:
            await self.prisma_client.db.litellm_managedobjecttable.update(
                where={"id": job.id}, data=update_data
            )
            return True
        finalized_count = await self.prisma_client.db.litellm_managedobjecttable.update_many(
            where={"id": job.id, "status": "pricing", "batch_processed": True},
            data=update_data,
        )
        if finalized_count != 1:
            verbose_proxy_logger.warning(
                f"CheckBatchCost: finalize skipped for job {job.id} — the claim was reclaimed by another worker"
            )
            return False
        return True

    @staticmethod
    def _batch_cost_call_id(batch_id: str) -> str:
        """Deterministic, prefixed spend id for a batch — the prefix is what
        get_spend_logs_id keys on to use it as the SpendLogs request_id."""
        import uuid as _stdlib_uuid

        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            BATCH_COST_CALL_ID_PREFIX,
        )

        return BATCH_COST_CALL_ID_PREFIX + str(
            _stdlib_uuid.uuid5(_stdlib_uuid.NAMESPACE_URL, f"litellm:batch-cost:{batch_id}")
        )

    async def _spend_already_recorded(self, batch_id: str, job: Any = None) -> bool:
        """True when this batch's spend side effects already ran — a prior
        worker billed it (then died before finalizing, or lost its claim to
        reclamation). Spend side effects (spend log AND the key/team/daily
        counters, which increment independently of the spend log's primary
        key) must not run twice (codex P1 round 4).

        Two independent signals, either suffices:
        1. The row-local marker stamped by _mark_spend_recorded — works even
           with disable_spend_logs, where no SpendLogs row ever exists
           (codex P1 round 5).
        2. A SpendLogs row under the deterministic request_id.

        Errors on the SpendLogs lookup return False: with the claim held, the
        deterministic request_id still dedups the spend LOG on the retry —
        only the counters ride on this pre-check, and skipping billing on a
        transient read error would risk never billing at all."""
        if (
            job is not None
            and self._get_job_file_object(job).get(SPEND_RECORDED_MARKER_KEY) is True
        ):
            return True
        try:
            existing = await self.prisma_client.db.litellm_spendlogs.find_unique(
                where={"request_id": self._batch_cost_call_id(batch_id)}
            )
            return existing is not None
        except Exception as lookup_err:
            verbose_proxy_logger.error(
                f"CheckBatchCost: spend-already-recorded lookup failed for {batch_id}: {lookup_err}"
            )
            return False

    async def _mark_spend_recorded(self, job: Any) -> None:
        """Stamp the billing row the moment spend side effects have run, so
        the dedup pre-check works without a SpendLogs row (disable_spend_logs
        deployments — codex P1 round 5). Fenced to the claim this worker
        holds, matching release/finalize; a slow ex-owner whose claim was
        reclaimed writes 0 rows (that 2h-reclaim-vs-active-worker race is the
        pre-existing accepted residual, covered by the SpendLogs backstop
        when spend logs are enabled). Best effort: spend was already emitted,
        so a failed marker write must not release the claim or block
        finalization. Legacy schemas skip it: without the batch_processed
        column there is no claim reclamation, so the re-bill scenario the
        marker guards against cannot occur."""
        if not self._has_batch_processed_column:
            return
        try:
            stamped = dict(self._get_job_file_object(job))
            stamped[SPEND_RECORDED_MARKER_KEY] = True
            await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={"id": job.id, "status": "pricing", "batch_processed": True},
                data={"file_object": json.dumps(stamped)},
            )
        except Exception as mark_err:
            verbose_proxy_logger.critical(
                f"CheckBatchCost: could not stamp spend-recorded marker on job {job.id}; "
                f"if this worker dies before finalizing and spend logs are disabled, "
                f"reclamation may re-bill this batch: {mark_err}"
            )

    async def _reclaim_abandoned_pricing_claims(self) -> None:
        """Requeue rows stuck in a pricing claim: a worker that died between
        claiming and finalizing leaves batch_processed=True,status='pricing',
        which the primary query excludes forever — the batch would never be
        billed (codex P1 round 2). Claims older than the reclaim window are
        conditionally flipped back; if the dead worker DID write spend before
        dying, the deterministic per-batch litellm_call_id makes the re-priced
        spend row collide instead of double-billing."""
        if not self._has_batch_processed_column:
            return
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=ABANDONED_PRICING_CLAIM_RECLAIM_SECONDS
            )
            reclaimed = await self.prisma_client.db.litellm_managedobjecttable.update_many(
                where={
                    "file_purpose": "batch",
                    "status": "pricing",
                    "batch_processed": True,
                    "updated_at": {"lt": cutoff},
                },
                data={"batch_processed": False, "status": "validating"},
            )
            if reclaimed:
                verbose_proxy_logger.warning(
                    f"CheckBatchCost: reclaimed {reclaimed} abandoned pricing claim(s) for retry"
                )
        except Exception as reclaim_err:
            verbose_proxy_logger.error(
                f"CheckBatchCost: abandoned-claim reclaim failed: {reclaim_err}"
            )

    @staticmethod
    def _finalized_file_object(
        job: Any, response: "LiteLLMBatch", spend_recorded: bool = False
    ) -> str:
        """The provider response JSON with the registration stash re-attached:
        finalization must not discard litellm_attribution — the upstream
        ownership check matches on it, so dropping it would 404 key-only
        callers on their own batch after billing (codex P2). spend_recorded
        stamps the row-local dedup marker (also preserved from the stash) so
        finalization never erases evidence that billing ran."""
        finalized: Dict[str, Any] = json.loads(response.model_dump_json())
        stash = CheckBatchCost._get_job_file_object(job)
        for preserved_key in (
            "litellm_attribution",
            "model",
            "mixed_models",
            "total_records",
            SPEND_RECORDED_MARKER_KEY,
        ):
            if preserved_key in stash and (
                preserved_key not in finalized or finalized.get(preserved_key) is None
            ):
                finalized[preserved_key] = stash.get(preserved_key)
        if spend_recorded:
            finalized[SPEND_RECORDED_MARKER_KEY] = True
        return json.dumps(finalized)

    async def _get_user_info(self, batch_id, user_id) -> dict:
        """
        Look up user email and key alias by user_id for enriching the S3 callback metadata.
        Returns a dict with user_api_key_user_email and user_api_key_alias (both may be None).
        """
        try:
            user_row = await self.prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}
            )
            if user_row is None:
                return {}
            return {
                "user_api_key_user_email": getattr(user_row, "user_email", None),
                "user_api_key_alias": getattr(user_row, "user_alias", None),
            }
        except Exception as e:
            verbose_proxy_logger.error(
                f"CheckBatchCost: could not look up user {user_id} for batch {batch_id}: {e}"
            )
            return {}

    async def _cleanup_stale_managed_objects(self) -> None:
        """
        Mark managed objects older than MANAGED_OBJECT_STALENESS_CUTOFF_DAYS days
        in non-terminal states as 'stale_expired'. These will never complete and
        should not be polled.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS
        )
        result = await self.prisma_client.db.litellm_managedobjecttable.update_many(
            where={
                "file_purpose": "batch",
                "status": {
                    "not_in": [
                        "completed",
                        "complete",
                        "failed",
                        "expired",
                        "cancelled",
                        "stale_expired",
                        # claim-held rows belong to the reclaim sweep, never
                        # stale cleanup (codex P1 round 3)
                        "pricing",
                    ]
                },
                "created_at": {"lt": cutoff},
            },
            data={"status": "stale_expired"},
        )
        if result > 0:
            verbose_proxy_logger.warning(
                f"CheckBatchCost: marked {result} stale managed objects "
                f"(older than {MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days) as stale_expired"
            )

    async def _fallback_find_jobs(self) -> list:
        """Query batch jobs without the batch_processed filter (for older schemas)."""
        return await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "file_purpose": "batch",
                "status": {
                    "not_in": [
                        "failed",
                        "expired",
                        "cancelled",
                        "complete",
                        "completed",
                        "stale_expired",
                    ]
                },
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )

    @staticmethod
    def _record_error(
        prom_logger: Optional["PrometheusLogger"], error_type: str
    ) -> None:
        if prom_logger is not None:
            prom_logger.record_check_batch_cost_error(error_type)

    def _resolve_job_routing(
        self,
        job: "LiteLLM_ManagedObjectTable",
        prom_logger: Optional["PrometheusLogger"],
    ) -> Optional[Tuple[str, str]]:
        """
        Resolve (model_id, batch_id) for a managed-object row, where model_id is a router
        deployment id and batch_id is the raw provider batch id.

        Managed batches encode both in a base64 unified id. Unmanaged batches (created outside
        LiteLLM's own /v1/batches with a raw input_file_id) store the raw provider job id as
        unified_object_id instead; when track_unmanaged_batch_cost is enabled the model is derived
        from the provider-specific input_file_id layout (Vertex gs:// or Bedrock s3://) and mapped
        to a matching deployment. Returns None (recording a metric) when the row can't be routed.
        """
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            get_batch_id_from_unified_batch_id,
            get_model_id_from_unified_batch_id,
        )

        unified_object_id = job.unified_object_id
        decoded = _is_base64_encoded_unified_file_id(unified_object_id)
        if decoded:
            model_id = get_model_id_from_unified_batch_id(decoded)
            if model_id is None:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} because it is not a valid model id"
                )
                self._record_error(prom_logger, "invalid_model_id")
                return None
            return model_id, get_batch_id_from_unified_batch_id(decoded)

        if self._track_unmanaged_batch_cost:
            from litellm.llms.bedrock.batches.transformation import (
                BedrockBatchesConfig,
            )
            from litellm.llms.vertex_ai.batches.transformation import (
                VertexAIBatchTransformation,
            )

            input_file_id = self._get_input_file_id(job)
            if VertexAIBatchTransformation.is_unmanaged_gcs_batch_input_file_id(
                input_file_id
            ):
                assert input_file_id is not None  # narrowed by is_unmanaged_gcs_batch_input_file_id
                return self._resolve_unmanaged_provider_routing(
                    job=job,
                    prom_logger=prom_logger,
                    llm_provider="vertex_ai",
                    bare_model_name=VertexAIBatchTransformation.get_bare_model_name_from_gcs_file(
                        input_file_id
                    ),
                )
            if BedrockBatchesConfig.is_unmanaged_s3_batch_input_file_id(input_file_id):
                assert input_file_id is not None  # narrowed by is_unmanaged_s3_batch_input_file_id
                return self._resolve_unmanaged_provider_routing(
                    job=job,
                    prom_logger=prom_logger,
                    llm_provider="bedrock",
                    bare_model_name=BedrockBatchesConfig.get_bare_model_name_from_s3_file(
                        input_file_id
                    ),
                )
            verbose_proxy_logger.info(
                f"Skipping job {unified_object_id}: not a recognized unmanaged batch "
                "(no gs:// or s3:// input_file_id with an embedded model)"
            )
            self._record_error(prom_logger, "invalid_unified_id")
            return None

        verbose_proxy_logger.info(
            f"Skipping job {unified_object_id} because it is not a valid unified object id"
        )
        self._record_error(prom_logger, "invalid_unified_id")
        return None

    def _resolve_unmanaged_provider_routing(
        self,
        job: "LiteLLM_ManagedObjectTable",
        prom_logger: Optional["PrometheusLogger"],
        llm_provider: str,
        bare_model_name: str,
    ) -> Optional[Tuple[str, str]]:
        deployment_id = self._get_deployment_id_for_bare_model(bare_model_name, llm_provider)
        if deployment_id is None:
            verbose_proxy_logger.info(
                f"Skipping unmanaged {llm_provider} batch {job.unified_object_id}: no {llm_provider} "
                f"deployment configured for model {bare_model_name}"
            )
            self._record_error(prom_logger, "unmanaged_no_matching_deployment")
            return None

        return deployment_id, job.unified_object_id

    def _get_deployment_id_for_bare_model(
        self, bare_model_name: str, llm_provider: str
    ) -> Optional[str]:
        model_group = self.llm_router.resolve_model_name_from_model_id(bare_model_name)
        deployment_id = (
            self._get_deployment_id_for_provider(model_group, llm_provider) if model_group else None
        )
        if deployment_id is not None:
            return deployment_id

        return self._get_deployment_id_from_matching_deployments(
            bare_model_name, llm_provider
        )

    def _get_deployment_id_from_matching_deployments(
        self, bare_model_name: str, llm_provider: str
    ) -> Optional[str]:
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        for deployment in self.llm_router.get_model_list(model_name=None) or []:
            litellm_params = deployment.get("litellm_params") or {}
            actual_model = litellm_params.get("model")
            if not isinstance(actual_model, str):
                continue
            if not self._is_bare_model_match(actual_model, bare_model_name):
                continue
            try:
                _, deployment_llm_provider, _, _ = get_llm_provider(
                    model=actual_model,
                    custom_llm_provider=litellm_params.get("custom_llm_provider"),
                )
            except Exception:
                continue
            if deployment_llm_provider != llm_provider:
                continue
            model_info = deployment.get("model_info") or {}
            deployment_id = model_info.get("id")
            if isinstance(deployment_id, str):
                return deployment_id
        return None

    @staticmethod
    def _is_bare_model_match(actual_model: str, bare_model_name: str) -> bool:
        # Bedrock model ids may have ":" replaced with "-" in the S3 object key (see
        # BedrockBatchesConfig.get_bare_model_name_from_s3_file), so normalize both sides;
        # a no-op for providers like vertex_ai whose model ids never contain a colon.
        normalized_actual = actual_model.replace(":", "-")
        normalized_bare = bare_model_name.replace(":", "-")
        return (
            normalized_actual == normalized_bare
            or normalized_actual.endswith(f"/{normalized_bare}")
        )

    def _get_deployment_id_for_provider(
        self, model_group: str, llm_provider: str
    ) -> Optional[str]:
        """
        Returns the first deployment id for `model_group` whose provider is `llm_provider`,
        skipping deployments from other providers that happen to share the model group name.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        for deployment_id in self.llm_router.get_model_ids(model_name=model_group):
            deployment_info = self.llm_router.get_deployment(model_id=deployment_id)
            if deployment_info is None:
                continue
            try:
                _, deployment_llm_provider, _, _ = get_llm_provider(
                    model=deployment_info.litellm_params.model,
                    custom_llm_provider=deployment_info.litellm_params.custom_llm_provider,
                )
            except Exception:
                continue
            if deployment_llm_provider == llm_provider:
                return deployment_id
        return None

    @staticmethod
    def _get_input_file_id(job: "LiteLLM_ManagedObjectTable") -> Optional[str]:
        import json

        from litellm.types.utils import LiteLLMBatch

        file_object = job.file_object
        if isinstance(file_object, str):
            try:
                file_object = json.loads(file_object)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(file_object, dict):
            return None
        try:
            return LiteLLMBatch.model_validate(file_object).input_file_id
        except Exception:
            return None

    async def _track_completed_batch_cost(
        self,
        job: "LiteLLM_ManagedObjectTable",
        response: "LiteLLMBatch",
        model_id: str,
        batch_id: str,
        prom_logger: Optional["PrometheusLogger"],
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Fetch a completed batch's results, compute cost/usage, and emit the
        aretrieve_batch spend log. Returns (model_name, llm_provider) on
        success, None when the job can't be routed to a deployment. Raises on
        results-fetch or cost-computation failures so the caller can leave the
        job unprocessed and retry it on a later poll.
        """
        from litellm.batches.batch_utils import (
            _get_file_content_as_dictionary,
            calculate_batch_cost_and_usage,
        )
        from litellm.files.main import afile_content
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
        )

        verbose_proxy_logger.info(
            f"Batch ID: {batch_id} is complete, tracking cost and usage"
        )

        # aretrieve_batch is called with the raw provider batch ID, so response.id
        # is the raw provider value (e.g. "batch_20260223-0518.234"). We need the
        # unified base64 ID in the S3 log so downstream consumers can correlate it
        # back to the batch they submitted via the proxy.
        #
        # CheckBatchCost builds its own LiteLLMLogging object (logging_obj below) and
        # calls async_success_handler(result=response) directly. That handler calls
        # _build_standard_logging_payload(response, ...) which reads response.id at
        # that point — so setting response.id here is sufficient.
        #
        # The HTTP endpoint does this substitution via the managed files hook
        # (async_post_call_success_hook). CheckBatchCost bypasses that hook entirely,
        # so we do it explicitly here.
        response.id = job.unified_object_id

        # This background job runs as default_user_id, so going through the HTTP endpoint
        # would trigger check_managed_file_id_access and get 403. Instead, extract the raw
        # provider file ID and call afile_content directly with deployment credentials.
        raw_output_file_id = response.output_file_id
        decoded = _is_base64_encoded_unified_file_id(raw_output_file_id)
        if decoded:
            try:
                raw_output_file_id = decoded.split("llm_output_file_id,")[1].split(";")[0]
            except (IndexError, AttributeError):
                pass

        credentials = self.llm_router.get_deployment_credentials_with_provider(model_id) or {}
        _file_content = await afile_content(
            file_id=raw_output_file_id,
            **credentials,
        )

        # Access content - handle both direct attribute and method call
        if hasattr(_file_content, 'content'):
            content_bytes = _file_content.content  # type: ignore[union-attr]
        elif hasattr(_file_content, 'read'):
            content_bytes = await _file_content.read()  # type: ignore[misc]
        else:
            content_bytes = _file_content  # type: ignore[assignment]

        file_content_as_dict = _get_file_content_as_dictionary(
            content_bytes  # type: ignore[arg-type]
        )

        # Record output file size
        if prom_logger and content_bytes:
            try:
                prom_logger.record_managed_file_size(
                    size_bytes=len(content_bytes),  # type: ignore
                    purpose="batch",
                    file_type="output",
                    model=model_id,
                )
            except Exception:
                pass

        deployment_info = self.llm_router.get_deployment(model_id=model_id)
        if deployment_info is None:
            verbose_proxy_logger.info(
                f"Skipping job {job.unified_object_id} because it is not a valid deployment info"
            )
            self._record_error(prom_logger, "deployment_not_found")
            return None
        custom_llm_provider = deployment_info.litellm_params.custom_llm_provider
        litellm_model_name = deployment_info.litellm_params.model

        model_name, llm_provider, _, _ = get_llm_provider(
            model=litellm_model_name,
            custom_llm_provider=custom_llm_provider,
        )

        # Rows registered by /v1/messages/batches stash the batch's actual
        # client-facing model. The routing deployment may be a borrowed
        # same-provider one (shared workspace credentials), so prefer the
        # stashed model for the spend log's model attribution — pricing for
        # anthropic rows already uses each result row's own model field.
        stashed_file_object = self._get_job_file_object(job)
        if stashed_file_object.get("litellm_attribution") and stashed_file_object.get("model"):
            model_name = str(stashed_file_object["model"])
        # Mixed-model batches: the registering deployment is an arbitrary
        # same-provider one, so neither its model_name nor its custom batch
        # rates may drive pricing — strip both and let every result row price
        # by its own model field (codex P1).
        stashed_mixed_models = bool(stashed_file_object.get("mixed_models"))
        if stashed_mixed_models:
            model_name = None

        # CheckBatchCost bypasses async_post_call_success_hook, so convert raw
        # output/error file IDs to managed base64 IDs before the DB write here.
        managed_files_hook = self.proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_hook is not None:
            from litellm.proxy._types import UserAPIKeyAuth
            _minimal_auth = UserAPIKeyAuth(
                user_id=job.created_by or "default-user-id",
                team_id=getattr(job, "team_id", None),
            )
            for _file_attr in ["output_file_id", "error_file_id"]:
                _raw_file_id = getattr(response, _file_attr, None)
                if _raw_file_id and not _is_base64_encoded_unified_file_id(_raw_file_id):
                    try:
                        _unified_file_id = managed_files_hook.get_unified_output_file_id(
                            output_file_id=_raw_file_id,
                            model_id=model_id,
                            model_name=str(model_name) if model_name else deployment_info.model_name or None,
                        )
                        await managed_files_hook.store_unified_file_id(
                            file_id=_unified_file_id,
                            file_object=None,
                            litellm_parent_otel_span=None,
                            model_mappings={model_id: _raw_file_id},
                            user_api_key_dict=_minimal_auth,
                        )
                        setattr(response, _file_attr, _unified_file_id)
                        verbose_proxy_logger.info(
                            f"CheckBatchCost: converted {_file_attr} "
                            f"{_raw_file_id!r} -> managed ID for batch {batch_id}"
                        )
                    except Exception as _e:
                        verbose_proxy_logger.warning(
                            f"CheckBatchCost: failed to create managed file ID for "
                            f"{_file_attr}={_raw_file_id!r}: {_e}"
                        )

        # Pass deployment model_info so custom batch pricing
        # (input_cost_per_token_batches etc.) is used for cost calc.
        # Mixed-model rows deliberately drop it (see stashed_mixed_models).
        deployment_model_info = (
            {} if stashed_mixed_models else (deployment_info.model_info.model_dump() if deployment_info.model_info else {})
        )
        batch_cost, batch_usage, batch_models = (
            await calculate_batch_cost_and_usage(
                file_content_dictionary=file_content_as_dict,
                custom_llm_provider=llm_provider,  # type: ignore
                model_name=model_name,
                model_info=deployment_model_info,  # type: ignore[arg-type]
            )
        )
        logging_obj = LiteLLMLogging(
            model=batch_models[0] if batch_models else (model_name or "unknown"),
            messages=[{"role": "user", "content": "<retrieve_batch>"}],
            stream=False,
            call_type="aretrieve_batch",
            start_time=datetime.now(),
            # Deterministic per-batch id, honored as the SpendLogs request_id
            # by get_spend_logs_id via its prefix (aretrieve_batch normally
            # keys on a response hash, which the poller's freshly-minted
            # managed file ids change every retry). Combined with the
            # already-billed pre-check in the caller, retried pricing cannot
            # insert a second spend row (codex P1 round 4).
            litellm_call_id=self._batch_cost_call_id(batch_id),
            function_id=str(uuid.uuid4()),
        )

        creator_user_id = job.created_by
        user_info = await self._get_user_info(batch_id, job.created_by)

        # The table only carries created_by/team_id, so key-level (and
        # end-user) spend attribution needs the submitting key's identity.
        # Routes that create batches out-of-band (/v1/messages/batches) stash
        # it in file_object.litellm_attribution; rows without the stash keep
        # the previous user-only attribution. team_id threads through in
        # either case (it was silently dropped before, so batch spend never
        # reached team running/daily totals).
        attribution = self._get_job_attribution(job)
        metadata: Dict[str, Any] = {
            "user_api_key_user_id": creator_user_id,
            **user_info,
        }
        job_team_id = getattr(job, "team_id", None) or attribution.get("user_api_key_team_id")
        if job_team_id:
            metadata["user_api_key_team_id"] = job_team_id
        for attribution_key in ("user_api_key", "user_api_key_alias", "user_api_key_end_user_id"):
            if attribution.get(attribution_key):
                metadata[attribution_key] = attribution[attribution_key]

        logging_obj.update_environment_variables(
            litellm_params={
                # set the user-agent header so that S3 callback consumers can easily identify CheckBatchCost callbacks
                "proxy_server_request": {
                    "headers": {
                        "user-agent": CHECK_BATCH_COST_USER_AGENT,
                    }
                },
                "metadata": metadata,
            },
            optional_params={},
        )

        await logging_obj.async_success_handler(
            result=response,
            batch_cost=batch_cost,
            batch_usage=batch_usage,
            batch_models=batch_models,
        )

        # Record batch duration (completed_at - created_at)
        if prom_logger and response.completed_at and response.created_at:
            duration_seconds = float(response.completed_at - response.created_at)
            if duration_seconds >= 0:
                prom_logger.record_managed_batch_duration(
                    duration_seconds=duration_seconds,
                    model=model_name,
                    api_provider=str(llm_provider) if llm_provider else None,
                )

        return model_name, str(llm_provider) if llm_provider else None

    async def check_batch_cost(self):
        """
        Check if the batch JOB has been tracked.
        - get all status="validating" and file_purpose="batch" jobs
        - check if batch is now complete
        - if not, return False
        - if so, return True
        """
        try:
            from litellm.integrations.prometheus import PrometheusLogger

            prom_logger = PrometheusLogger.get_instance()
        except Exception as e:
            verbose_proxy_logger.error(
                f"CheckBatchCost: could not get Prometheus logger: {e}"
            )
            prom_logger = None

        processed_models: List[Tuple[Optional[str], Optional[str]]] = []

        try:
            # Reclaim FIRST: cleanup running first would stale_expire an
            # old claim-held row while batch_processed stays True — a state
            # the delete gate reads as finalized (codex P1 round 3).
            await self._reclaim_abandoned_pricing_claims()
            await self._cleanup_stale_managed_objects()
        except Exception as cleanup_err:
            verbose_proxy_logger.warning(
                f"CheckBatchCost: stale cleanup failed (poll will continue): {cleanup_err}"
            )

        # Look for all batches that have not yet been processed by CheckBatchCost.
        # self._has_batch_processed_column is cached after the first probe so that
        # older schemas don't pay a guaranteed-failing primary query + warning on
        # every subsequent poll cycle.
        if self._has_batch_processed_column:
            try:
                # Include "complete"/"completed" batches: the retrieve_batch
                # endpoint may transition a batch to "complete" before
                # CheckBatchCost runs.  The batch_processed=False filter
                # already prevents reprocessing finished batches.
                jobs = await self.prisma_client.db.litellm_managedobjecttable.find_many(
                    where={
                        "file_purpose": "batch",
                        "batch_processed": False,
                        "status": {
                            "not_in": [
                                "failed",
                                "expired",
                                "cancelled",
                                "stale_expired",
                            ]
                        },
                    },
                    take=MAX_OBJECTS_PER_POLL_CYCLE,
                    order={"created_at": "asc"},
                )
            except Exception as query_err:
                if (
                    "batch_processed" not in str(query_err).lower()
                    and "unknown column" not in str(query_err).lower()
                    and "does not exist" not in str(query_err).lower()
                ):
                    raise
                # Permanent schema gap — cache the result so future cycles skip straight to fallback
                self._has_batch_processed_column = False
                verbose_proxy_logger.warning(
                    "CheckBatchCost: batch_processed column not found, querying without it"
                )
                jobs = await self._fallback_find_jobs()
        else:
            jobs = await self._fallback_find_jobs()
        for job in jobs:
            routing = self._resolve_job_routing(job, prom_logger)
            if routing is None:
                continue
            model_id, batch_id = routing

            verbose_proxy_logger.info(
                f"Querying model ID: {model_id} for cost and usage of batch ID: {batch_id}"
            )

            try:
                response = await self.llm_router.aretrieve_batch(
                    model=model_id,
                    batch_id=batch_id,
                    litellm_metadata={
                        "user_api_key_user_id": job.created_by or "default-user-id",
                        "batch_ignore_default_logging": True,
                    },
                )
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Skipping job {job.unified_object_id} because of error querying model ID: {model_id} for cost and usage of batch ID: {batch_id}: {e}"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error(
                        "provider_retrieval_error"
                    )
                continue

            ## RETRIEVE THE BATCH JOB OUTPUT FILE
            # Terminal-but-not-completed jobs (stopped/failed/expired) can
            # still carry billable partial output — Bedrock writes whatever
            # records finished, and the batch route serves those results.
            # Finalizing them without pricing hands out free tokens via
            # cancel-after-partial-processing (codex P1); salvage-price when
            # an output object was predicted for the job.
            salvaged_output_file_id: Optional[str] = None
            if response.status in ("failed", "expired", "cancelled") and response.output_file_id is None:
                response_metadata = getattr(response, "metadata", None)
                if isinstance(response_metadata, dict) and response_metadata.get("output_file_uri"):
                    salvaged_output_file_id = str(response_metadata["output_file_uri"])
                    response.output_file_id = salvaged_output_file_id

            if response.output_file_id is not None and (
                response.status == "completed" or salvaged_output_file_id is not None
            ):
                terminal_status = "complete" if response.status == "completed" else response.status
                if not await self._claim_job(job):
                    # Another worker owns this row (or the claim errored) —
                    # never price without holding the claim.
                    continue
                if await self._spend_already_recorded(batch_id, job):
                    # A prior worker billed this batch but died before
                    # finalizing (or was reclaimed) — finalize WITHOUT
                    # re-running spend side effects.
                    verbose_proxy_logger.warning(
                        f"CheckBatchCost: spend already recorded for batch {batch_id}; "
                        f"finalizing job {job.id} without re-billing"
                    )
                    try:
                        already_billed_update: dict = {
                            "status": terminal_status,
                            "file_object": self._finalized_file_object(
                                job, response, spend_recorded=True
                            ),
                        }
                        if self._has_batch_processed_column:
                            already_billed_update["batch_processed"] = True
                        await self._finalize_job(job, already_billed_update)
                    except Exception as db_err:
                        verbose_proxy_logger.error(
                            f"CheckBatchCost: failed to finalize already-billed job {job.id}: {db_err}"
                        )
                    continue
                try:
                    tracked = await self._track_completed_batch_cost(
                        job=job,
                        response=response,
                        model_id=model_id,
                        batch_id=batch_id,
                        prom_logger=prom_logger,
                    )
                except Exception as tracking_err:
                    # S3-specific missing-object signatures ONLY: generic
                    # markers ("not found", "404") also match pricing errors
                    # like "Model not found in cost map" and would finalize
                    # billable partial output at zero (codex P1 round 3).
                    # Unrecognized errors retry — the reclaim sweep keeps
                    # retries alive, so the safe direction is to never
                    # zero-finalize on ambiguity.
                    error_text = str(tracking_err).lower()
                    output_definitively_missing = any(
                        marker in error_text
                        for marker in ("nosuchkey", "no such key", "specified key does not exist")
                    )
                    if salvaged_output_file_id is not None and output_definitively_missing:
                        # Salvage path, output object confirmed absent: the
                        # job died before writing any records — finalize
                        # without cost instead of retrying a fetch that can
                        # never succeed. Any OTHER error (transient S3,
                        # credentials, pricing) releases and retries — it
                        # must not zero out billable partial output (codex
                        # P1 round 2). The claim stays held; finalization
                        # happens below.
                        verbose_proxy_logger.warning(
                            f"CheckBatchCost: no salvageable output for terminal batch {batch_id} "
                            f"(job {job.id}, status {response.status}): {tracking_err}"
                        )
                        tracked = None
                    else:
                        await self._release_job_claim(job)
                        verbose_proxy_logger.error(
                            f"CheckBatchCost: failed to track cost for batch {batch_id} "
                            f"(job {job.id}); released the claim so the next poll retries: {tracking_err}"
                        )
                        self._record_error(prom_logger, "cost_tracking_error")
                        continue
                if tracked is None and salvaged_output_file_id is None:
                    # Unroutable row: release so a config fix can still bill it.
                    await self._release_job_claim(job)
                    continue

                # Track this job for the final metrics summary
                if tracked is not None:
                    processed_models.append(tracked)
                    # Spend side effects just ran — stamp the row before
                    # finalizing so a crash in between can't re-bill after
                    # reclamation, even with disable_spend_logs set.
                    await self._mark_spend_recorded(job)

                # finalize, fenced to the claim this worker holds
                try:
                    update_data: dict = {
                        "status": terminal_status,
                        "file_object": self._finalized_file_object(
                            job, response, spend_recorded=tracked is not None
                        ),
                    }
                    if self._has_batch_processed_column:
                        update_data["batch_processed"] = True
                    await self._finalize_job(job, update_data)
                except Exception as db_err:
                    verbose_proxy_logger.error(
                        f"CheckBatchCost: failed to mark job {job.id} {terminal_status} in DB: {db_err}"
                    )

            elif response.status in ("failed", "expired", "cancelled"):
                # Terminal with no output object at all — nothing to price.
                if not await self._claim_job(job):
                    continue
                try:
                    update_data = {
                        "status": response.status,
                        "file_object": self._finalized_file_object(job, response),
                    }
                    if self._has_batch_processed_column:
                        update_data["batch_processed"] = True
                    await self._finalize_job(job, update_data)
                    verbose_proxy_logger.info(
                        f"CheckBatchCost: marked job {job.id} as {response.status} in DB"
                    )
                except Exception as db_err:
                    verbose_proxy_logger.error(
                        f"CheckBatchCost: failed to mark job {job.id} as {response.status} in DB: {db_err}"
                    )

        # Record polling run metrics (always, even if nothing was processed)
        if prom_logger:
            prom_logger.record_check_batch_cost_run(
                jobs_polled=len(jobs),
                processed_models=processed_models if processed_models else None,
            )
