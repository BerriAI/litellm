"""
Polls LiteLLM_ManagedObjectTable to check if the batch job is complete, and if the cost has been tracked.
"""

from litellm._uuid import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from litellm._logging import verbose_proxy_logger

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

    async def check_batch_cost(self):
        """
        Check if the batch JOB has been tracked.
        - get all status="validating" and file_purpose="batch" jobs
        - check if batch is now complete
        - if not, return False
        - if so, return True
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
            get_batch_id_from_unified_batch_id,
            get_model_id_from_unified_batch_id,
        )

        # Look for all batches that have not yet been processed by CheckBatchCost
        jobs = await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "file_purpose": "batch",
                "batch_processed" : False,
                "status": {"not_in": ["failed", "expired", "cancelled"]}
            }
        )
        completed_jobs = []

        for job in jobs:
            # get the model from the job
            unified_object_id = job.unified_object_id
            decoded_unified_object_id = _is_base64_encoded_unified_file_id(
                unified_object_id
            )
            if not decoded_unified_object_id:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} because it is not a valid unified object id"
                )
                continue
            else:
                unified_object_id = decoded_unified_object_id

            model_id = get_model_id_from_unified_batch_id(unified_object_id)
            batch_id = get_batch_id_from_unified_batch_id(unified_object_id)

            if model_id is None:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} because it is not a valid model id"
                )
                continue

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
                    f"Skipping job {unified_object_id} because of error querying model ID: {model_id} for cost and usage of batch ID: {batch_id}: {e}"
                )
                continue

            ## RETRIEVE THE BATCH JOB OUTPUT FILE
            if (
                response.status == "completed"
                and response.output_file_id is not None
            ):
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
                    content_bytes = _file_content.content
                elif hasattr(_file_content, 'read'):
                    content_bytes = await _file_content.read()
                else:
                    content_bytes = _file_content

                file_content_as_dict = _get_file_content_as_dictionary(
                    content_bytes
                )

                deployment_info = self.llm_router.get_deployment(model_id=model_id)
                if deployment_info is None:
                    verbose_proxy_logger.info(
                        f"Skipping job {unified_object_id} because it is not a valid deployment info"
                    )
                    continue
                custom_llm_provider = deployment_info.litellm_params.custom_llm_provider
                litellm_model_name = deployment_info.litellm_params.model

                model_name, llm_provider, _, _ = get_llm_provider(
                    model=litellm_model_name,
                    custom_llm_provider=custom_llm_provider,
                )

                # Pass deployment model_info so custom batch pricing
                # (input_cost_per_token_batches etc.) is used for cost calc
                deployment_model_info = deployment_info.model_info.model_dump() if deployment_info.model_info else {}
                batch_cost, batch_usage, batch_models = (
                    await calculate_batch_cost_and_usage(
                        file_content_dictionary=file_content_as_dict,
                        custom_llm_provider=llm_provider,  # type: ignore
                        model_name=model_name,
                        model_info=deployment_model_info,
                    )
                )
                logging_obj = LiteLLMLogging(
                    model=batch_models[0],
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
                            }
                        },
                        "metadata": {
                            "user_api_key_user_id": creator_user_id,
                            **user_info,
                        },
                    },
                    optional_params={},
                )

                await logging_obj.async_success_handler(
                    result=response,
                    batch_cost=batch_cost,
                    batch_usage=batch_usage,
                    batch_models=batch_models,
                )

                # mark the job as complete
                completed_jobs.append(job)

            if len(completed_jobs) > 0:
                await self.prisma_client.db.litellm_managedobjecttable.update_many(
                    where={"id": {"in": [job.id for job in completed_jobs]}},
                    data={"batch_processed": True, "status": "complete"},
                )
