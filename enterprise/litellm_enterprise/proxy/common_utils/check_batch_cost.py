"""
Polls LiteLLM_ManagedObjectTable to check if the batch job is complete, and if the cost has been tracked.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router


CHECK_BATCH_COST_USER_AGENT = "LiteLLM Proxy/CheckBatchCost"


class CheckBatchCost:
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
        # Cached after the first poll cycle. Once we know the column is absent we skip
        # the guaranteed-failing primary query on every subsequent cycle.
        self._has_batch_processed_column: bool = True

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
            verbose_proxy_logger.error(f"CheckBatchCost: could not look up user {user_id} for batch {batch_id}: {e}")
            return {}

    async def _cleanup_stale_managed_objects(self) -> None:
        """
        Mark managed objects older than MANAGED_OBJECT_STALENESS_CUTOFF_DAYS days
        in non-terminal states as 'stale_expired'. These will never complete and
        should not be polled.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS)
        result = await self.prisma_client.db.litellm_managedobjecttable.update_many(
            where={
                "file_purpose": "batch",
                "status": {"not_in": ["completed", "complete", "failed", "expired", "cancelled", "stale_expired"]},
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

    async def check_batch_cost(self):
        """
        Check if the batch JOB has been tracked.
        - get all status="validating" and file_purpose="batch" jobs
        - check if batch is now complete
        - if not, return False
        - if so, return True
        """
        verbose_proxy_logger.warning("CheckBatchCost: Start==================")

        from litellm.batches.batch_utils import (
            _get_file_content_as_dictionary,
            calculate_batch_cost_and_usage,
        )
        from litellm.files.main import afile_content
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            get_batch_id_from_unified_batch_id,
            get_model_id_from_unified_batch_id,
            decode_data_from_unified_file_id,
        )

        try:
            from litellm.integrations.prometheus import PrometheusLogger
            prom_logger = PrometheusLogger.get_instance()
        except Exception as e:
            verbose_proxy_logger.error(f"CheckBatchCost: could not get Prometheus logger: {e}")
            prom_logger = None

        processed_models: List[Tuple[Optional[str], Optional[str]]] = []

        try:
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
                if "batch_processed" not in str(query_err).lower() and "unknown column" not in str(query_err).lower() and "does not exist" not in str(query_err).lower():
                    raise
                # Permanent schema gap — cache the result so future cycles skip straight to fallback
                self._has_batch_processed_column = False
                verbose_proxy_logger.warning(
                    "CheckBatchCost: batch_processed column not found, querying without it"
                )
                jobs = await self._fallback_find_jobs()
        else:
            jobs = await self._fallback_find_jobs()
        verbose_proxy_logger.debug(f"CheckBatchCost: Found {len(jobs)} jobs to process")
        for job in jobs:
            # get the model from the job
            unified_object_id = job.unified_object_id
            verbose_proxy_logger.warning("Processing job with unified_object_id: %s", unified_object_id)
            decoded_unified_object_id = _is_base64_encoded_unified_file_id(
                unified_object_id
            )
            verbose_proxy_logger.debug(f"CheckBatchCost: Decoded unified_object_id: {decoded_unified_object_id!r}")
            if not decoded_unified_object_id:
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} because it is not a valid unified object id"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error("invalid_unified_id")
                continue
            else:
                unified_object_id = decoded_unified_object_id

            model_id = get_model_id_from_unified_batch_id(unified_object_id)
            batch_id = get_batch_id_from_unified_batch_id(unified_object_id)
            verbose_proxy_logger.debug(f"CheckBatchCost: Extracted model_id={model_id!r}, batch_id={batch_id!r} from unified_object_id")

            if model_id is None:
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} because it is not a valid model id"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error("invalid_model_id")
                continue

            verbose_proxy_logger.warning(
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
                verbose_proxy_logger.warning(
                    f"Skipping job {unified_object_id} because of error querying model ID: {model_id} for cost and usage of batch ID: {batch_id}: {e}"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error("provider_retrieval_error")
                continue

            ## RETRIEVE THE BATCH JOB OUTPUT FILE
            if (
                response.status == "completed"
                and response.output_file_id is not None
            ):
                tags=(decode_data_from_unified_file_id(unified_object_id, "tags") or "").split(",")
                if "batch" not in tags:
                    tags.append("batch")
                api_alias=decode_data_from_unified_file_id(unified_object_id, "api_alias")
                apihash=decode_data_from_unified_file_id(unified_object_id, "apihash")
                print(f"file: {unified_object_id}, tags={tags}, api_alias={api_alias}, apihash={apihash}")
                verbose_proxy_logger.warning(
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
                verbose_proxy_logger.debug(f"CheckBatchCost: Raw output_file_id={raw_output_file_id!r}")
                decoded = _is_base64_encoded_unified_file_id(raw_output_file_id)
                verbose_proxy_logger.debug(f"CheckBatchCost: Decoded output_file_id={decoded!r}")
                if decoded:
                    try:
                        raw_output_file_id = decoded.split("llm_output_file_id,")[1].split(";")[0]
                        verbose_proxy_logger.debug(f"CheckBatchCost: Extracted raw_output_file_id={raw_output_file_id!r} from decoded value")
                    except (IndexError, AttributeError) as parse_err:
                        verbose_proxy_logger.debug(f"CheckBatchCost: Failed to parse output_file_id from decoded: {parse_err}")

                credentials = self.llm_router.get_deployment_credentials_with_provider(model_id) or {}
                verbose_proxy_logger.debug(f"CheckBatchCost: Deployment credentials keys={list(credentials.keys()) if credentials else 'None'}")
                verbose_proxy_logger.debug(f"CheckBatchCost: Calling afile_content with file_id={raw_output_file_id!r}")
                _file_content = await afile_content(
                    file_id=raw_output_file_id,
                    **credentials,
                )
                verbose_proxy_logger.debug(f"CheckBatchCost: afile_content returned type={type(_file_content).__name__}")
                
                # Access content - handle both direct attribute and method call
                if hasattr(_file_content, 'content'):
                    content_bytes = _file_content.content  # type: ignore[union-attr]
                    verbose_proxy_logger.debug(f"CheckBatchCost: Got content from .content attribute, size={len(content_bytes) if content_bytes else 0} bytes")
                elif hasattr(_file_content, 'read'):
                    content_bytes = await _file_content.read()  # type: ignore[misc]
                    verbose_proxy_logger.debug(f"CheckBatchCost: Got content from .read() method, size={len(content_bytes) if content_bytes else 0} bytes")
                else:
                    content_bytes = _file_content  # type: ignore[assignment]
                    verbose_proxy_logger.debug(f"CheckBatchCost: Got content directly, type={type(content_bytes).__name__}, size={len(content_bytes) if content_bytes else 0} bytes")

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
                verbose_proxy_logger.debug(f"CheckBatchCost: get_deployment(model_id={model_id!r}) returned: {deployment_info is not None}")
                if deployment_info is None:
                    verbose_proxy_logger.warning(
                        f"Skipping job {unified_object_id} because it is not a valid deployment info"
                    )
                    if prom_logger:
                        prom_logger.record_check_batch_cost_error("deployment_not_found")
                    continue
                
                custom_llm_provider = deployment_info.litellm_params.custom_llm_provider
                litellm_model_name = deployment_info.litellm_params.model
                verbose_proxy_logger.debug(f"CheckBatchCost: deployment_info - custom_llm_provider={custom_llm_provider!r}, litellm_params.model={litellm_model_name!r}, model_name={deployment_info.model_name!r}")

                model_name, llm_provider, _, _ = get_llm_provider(
                    model=litellm_model_name,
                    custom_llm_provider=custom_llm_provider,
                )
                verbose_proxy_logger.debug(f"CheckBatchCost: get_llm_provider result - model_name={model_name!r}, llm_provider={llm_provider!r}")

                # CheckBatchCost bypasses async_post_call_success_hook, so convert raw
                # output/error file IDs to managed base64 IDs before the DB write here.
                managed_files_hook = self.proxy_logging_obj.get_proxy_hook("managed_files")
                if managed_files_hook is not None:
                    from litellm.proxy._types import UserAPIKeyAuth
                    _minimal_auth = UserAPIKeyAuth(
                        key_alias=api_alias,
                        api_key=apihash,
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
                                verbose_proxy_logger.warning(
                                    f"CheckBatchCost: converted {_file_attr} "
                                    f"{_raw_file_id!r} -> managed ID for batch {batch_id}"
                                )
                            except Exception as _e:
                                verbose_proxy_logger.warning(
                                    f"CheckBatchCost: failed to create managed file ID for "
                                    f"{_file_attr}={_raw_file_id!r}: {_e}"
                                )

                # Pass deployment model_info so custom batch pricing
                # (input_cost_per_token_batches etc.) is used for cost calc
                deployment_model_info = deployment_info.model_info.model_dump() if deployment_info.model_info else {}
                model_name=deployment_info.model_name
                verbose_proxy_logger.debug(f"CheckBatchCost: deployment_model_info keys={list(deployment_model_info.keys()) if deployment_model_info else 'empty'}")
                verbose_proxy_logger.debug(f"CheckBatchCost: Calling calculate_batch_cost_and_usage with custom_llm_provider={llm_provider!r}, model_name={model_name!r}, file_content_dict_entries={len(file_content_as_dict) if file_content_as_dict else 0}")
                batch_cost, batch_usage, batch_models = (
                    await calculate_batch_cost_and_usage(
                        file_content_dictionary=file_content_as_dict,
                        custom_llm_provider=llm_provider,  # type: ignore
                        model_name=model_name,
                        model_info=deployment_model_info,  # type: ignore[arg-type]
                    )
                )
                verbose_proxy_logger.debug(f"CheckBatchCost: calculate_batch_cost_and_usage result - batch_cost={batch_cost}, batch_usage={batch_usage}, batch_models={batch_models}")
                logging_obj = LiteLLMLogging(
                    model=model_name,
                    messages=[{"role": "user", "content": "<retrieve_batch>"}],
                    stream=False,
                    call_type="aretrieve_batch",
                    start_time=datetime.now(),
                    litellm_call_id=str(uuid.uuid4()),
                    function_id=str(uuid.uuid4()),
                )

                creator_user_id = job.created_by
                user_info = await self._get_user_info(batch_id, job.created_by)
                

                logging_obj.update_environment_variables(
                    litellm_params={
                        # set the user-agent header so that S3 callback consumers can easily identify CheckBatchCost callbacks
                        "proxy_server_request": {
                            "headers": {
                                "user-agent": CHECK_BATCH_COST_USER_AGENT,
                                "x-litellm-tags": ",".join(tags),
                            }
                        },
                        "metadata": {
                            "user_api_key_user_id": creator_user_id,
                            **user_info,
                            # "custom_llm_provider": llm_provider,                            
                            # "llm_provider": llm_provider,
                            "user_api_key_alias": api_alias,
                            "user_api_key":apihash,
                            "tags":tags,
                            "request_tags": tags,

                        },
                    },
                    optional_params={},
                    # litellm_provider=llm_provider,
                    request_tags=tags,
                    tags=tags,
                    custom_llm_provider=llm_provider,
                )
                if apihash:
                    await logging_obj.async_success_handler(
                    result=response,
                    batch_cost=batch_cost,
                    batch_usage=batch_usage,
                    batch_models=[model_name],
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

                # Track this job for the final metrics summary
                processed_models.append((model_name, str(llm_provider) if llm_provider else None))

                # mark the job as complete
                try:
                    update_data: dict = {
                        "status": "complete",
                        "file_object": response.model_dump_json(),
                    }
                    if self._has_batch_processed_column:
                        update_data["batch_processed"] = True
                    verbose_proxy_logger.debug(f"CheckBatchCost: Updating job {job.id} with status=complete, batch_processed={self._has_batch_processed_column}")
                    await self.prisma_client.db.litellm_managedobjecttable.update(
                        where={"id": job.id},
                        data=update_data,
                    )
                    verbose_proxy_logger.debug(f"CheckBatchCost: Successfully marked job {job.id} as complete")
                except Exception as db_err:
                    verbose_proxy_logger.error(
                        f"CheckBatchCost: failed to mark job {job.id} complete in DB: {db_err}"
                    )

        # Record polling run metrics (always, even if nothing was processed)
        if prom_logger:
            prom_logger.record_check_batch_cost_run(
                jobs_polled=len(jobs),
                processed_models=processed_models if processed_models else None,
            )
