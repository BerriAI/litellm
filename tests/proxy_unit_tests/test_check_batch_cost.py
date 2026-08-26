"""
Unit tests for CheckBatchCost class.
Covers: stale-row cleanup (file_purpose scoping), paginated find_many,
the batch_processed-column fallback query, and routing of unmanaged
Vertex (raw gs:// input_file_id) and Bedrock (raw s3:// input_file_id,
ARN unified_object_id) batches with no managed unified id.
"""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_IS_B64 = "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id"
_CLAIM_UNIFIED_BATCH_ID = "dW5pZmllZF9iYXRjaF9pZA=="
_CLAIM_OUTPUT_FILE_ID = "file-output-123"


def _unmanaged_vertex_file_object(
    input_file_id="gs://bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-flash/abc.jsonl",
    status="validating",
):
    """A LiteLLMBatch JSON blob shaped like what the managed-files hook stores for an
    unmanaged Vertex batch (raw gs:// input_file_id)."""
    from litellm.types.utils import LiteLLMBatch

    return LiteLLMBatch(
        id="8823717160934178816",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        object="batch",
        status=status,
    ).model_dump_json()


def _unmanaged_bedrock_file_object(
    input_file_id=(
        "s3://bucket/litellm-bedrock-files-us.anthropic.claude-sonnet-4-20250514-v1-0"
        "-74b61828-9191-4d80-addb-5a0f9ab0ec6a.jsonl"
    ),
    status="validating",
):
    """A LiteLLMBatch JSON blob shaped like what gets stored for an unmanaged Bedrock
    batch (raw s3:// input_file_id, ARN unified_object_id)."""
    from litellm.types.utils import LiteLLMBatch

    return LiteLLMBatch(
        id="arn:aws:bedrock:us-east-1:298249409318:model-invocation-job/1ofb47x17jua",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        object="batch",
        status=status,
    ).model_dump_json()


class TestCheckBatchCost:
    """Test suite for CheckBatchCost class"""

    @pytest.fixture
    def mock_prisma_client(self):
        client = MagicMock()
        client.db = MagicMock()
        client.db.litellm_managedobjecttable = MagicMock()
        client.db.litellm_usertable = MagicMock()
        return client

    @pytest.fixture
    def mock_proxy_logging_obj(self):
        mock = MagicMock()
        mock.get_proxy_hook.return_value = None
        return mock

    @pytest.fixture
    def mock_llm_router(self):
        return MagicMock()

    @pytest.fixture
    def check_batch_cost_instance(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )

        return CheckBatchCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

    @pytest.mark.asyncio
    async def test_cleanup_scoped_to_batch_file_purpose(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """_cleanup_stale_managed_objects scopes its update to file_purpose='batch' only."""
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        # Return empty so the main poll loop exits immediately
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        await check_batch_cost_instance.check_batch_cost()

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        stale_call = calls[0]
        assert stale_call[1]["data"] == {"status": "stale_expired"}
        where = stale_call[1]["where"]
        assert where["file_purpose"] == "batch"
        assert "stale_expired" in where["status"]["not_in"]
        assert "created_at" in where

    @pytest.mark.asyncio
    async def test_startup_probe_confirms_batch_processed_support(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        mock_prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)

        await check_batch_cost_instance.confirm_batch_processed_support()

        probe_where = mock_prisma_client.db.litellm_managedobjecttable.find_first.call_args[1]["where"]
        assert probe_where["batch_processed"] is False
        assert check_batch_cost_instance.batch_processed_support_confirmed is True
        assert check_batch_cost_instance._has_batch_processed_column is True

    @pytest.mark.asyncio
    async def test_startup_probe_marks_column_absent(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        mock_prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(
            side_effect=Exception("column batch_processed does not exist")
        )

        await check_batch_cost_instance.confirm_batch_processed_support()

        assert check_batch_cost_instance.batch_processed_support_confirmed is False
        assert check_batch_cost_instance._has_batch_processed_column is False

    @pytest.mark.asyncio
    async def test_startup_probe_transient_error_defers_to_poll_cycle(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        mock_prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(
            side_effect=Exception("connection reset by peer")
        )

        await check_batch_cost_instance.confirm_batch_processed_support()

        assert check_batch_cost_instance.batch_processed_support_confirmed is False
        assert check_batch_cost_instance._has_batch_processed_column is True

    @pytest.mark.asyncio
    async def test_find_many_uses_pagination_and_excludes_stale(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """find_many is called with take, order, and all terminal statuses excluded."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        await check_batch_cost_instance.check_batch_cost()

        find_call = mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args
        assert find_call[1]["take"] == MAX_OBJECTS_PER_POLL_CYCLE
        assert find_call[1]["order"] == {"created_at": "asc"}
        not_in = find_call[1]["where"]["status"]["not_in"]
        assert "stale_expired" in not_in
        # "complete"/"completed" are intentionally NOT excluded from the
        # primary query — the batch_processed=False filter is sufficient.
        # This allows CheckBatchCost to pick up batches that were
        # transitioned to "complete" by the retrieve_batch endpoint
        # before CheckBatchCost had a chance to process them.
        assert "complete" not in not_in
        assert "completed" not in not_in
        assert find_call[1]["where"]["batch_processed"] is False
        assert check_batch_cost_instance.batch_processed_support_confirmed is True

    @pytest.mark.asyncio
    async def test_fallback_query_used_when_batch_processed_missing(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """Falls back to query without batch_processed when primary query raises."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        # First find_many (primary query) raises with a schema error; second (fallback) returns empty
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            side_effect=[Exception("column batch_processed does not exist"), []]
        )

        await check_batch_cost_instance.check_batch_cost()

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args_list
        )
        assert len(calls) == 2
        fallback_where = calls[1][1]["where"]
        assert "batch_processed" not in fallback_where
        assert "stale_expired" in fallback_where["status"]["not_in"]
        assert calls[1][1]["take"] == MAX_OBJECTS_PER_POLL_CYCLE
        # Column absence is now cached — next call should go straight to fallback
        assert check_batch_cost_instance._has_batch_processed_column is False
        assert check_batch_cost_instance.batch_processed_support_confirmed is False

    @pytest.mark.asyncio
    async def test_column_absence_cached_across_cycles(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """After column absence is discovered, subsequent cycles skip the primary query entirely."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        # Simulate column already known absent from a previous cycle
        check_batch_cost_instance._has_batch_processed_column = False
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        await check_batch_cost_instance.check_batch_cost()

        # Only one find_many call — the fallback directly, no primary query attempt
        assert (
            mock_prisma_client.db.litellm_managedobjecttable.find_many.call_count == 1
        )
        fallback_where = (
            mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args[1][
                "where"
            ]
        )
        assert "batch_processed" not in fallback_where

    @pytest.mark.asyncio
    async def test_fallback_completion_update_omits_batch_processed(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """When batch_processed column is absent, completion update must not include it.

        If it did, the update would fail silently, the job would never be marked done,
        and every subsequent poll cycle would re-log the cost (duplicate billing).
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-fallback-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        # Simulate column already known absent (e.g. discovered on a previous cycle)
        check_batch_cost_instance._has_batch_processed_column = False
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Build a fake batch response whose status triggers the completion branch
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "file-output-123"
        mock_response.model_dump_json.return_value = (
            '{"id":"batch-1","status":"completed"}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "openai"
        mock_deployment.litellm_params.model = "gpt-4"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gpt-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-4", "openai", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await check_batch_cost_instance.check_batch_cost()

        # The update must have been called — this is the core assertion.
        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        ), "Expected update() to be called exactly once for the completed job"
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert (
            "batch_processed" not in update_data
        ), "update() must NOT include batch_processed when column is absent"
        assert update_data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_output_fetch_passes_deployment_credentials_as_trusted_snapshot(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Bedrock resolves the output bucket ONLY from the immutable snapshot kwarg.

        Spreading the credentials as plain kwargs is not enough: get_litellm_params drops
        s3_bucket_name, so without _litellm_internal_model_credentials the cost poller
        cannot read the output file and every completed Bedrock batch stays unbilled.
        """
        from types import MappingProxyType
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        mock_job = MagicMock()
        mock_job.id = "job-bedrock-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "file-output-123"
        mock_response.model_dump_json.return_value = '{"id":"batch-1","status":"completed"}'

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={
                "custom_llm_provider": "bedrock",
                "s3_bucket_name": "configured-batch-bucket",
                "aws_region_name": "us-east-1",
            }
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "bedrock"
        mock_deployment.litellm_params.model = "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"recordId":"req-1"}'

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ) as mock_afile_content,
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"recordId": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(0.01, {"prompt_tokens": 10, "completion_tokens": 5}, ["claude-haiku-4-5"]),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", None, None),
            ),
            patch("litellm.litellm_core_utils.litellm_logging.Logging") as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await check_batch_cost_instance.check_batch_cost()

        mock_afile_content.assert_awaited()
        passed_kwargs = mock_afile_content.await_args[1]
        snapshot = passed_kwargs.get("_litellm_internal_model_credentials")
        assert snapshot is not None, "cost poller must pass the trusted credential snapshot"
        assert isinstance(
            snapshot, MappingProxyType
        ), "snapshot must be a MappingProxyType; a plain dict is rejected by get_configured_s3_bucket_name"
        assert snapshot["s3_bucket_name"] == "configured-batch-bucket"

    @pytest.mark.asyncio
    async def test_poller_prices_with_deployment_registered_batch_rates(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """The cost poller must price with the rates the router registered for the deployment.

        The deployment's raw model_info dict carries no litellm_params pricing, so passing
        its model_dump() made the poller bill custom-rate batches at the public cost-map
        price while the inline retrieve path billed the declared rate.
        """
        from unittest.mock import patch

        import litellm

        deployment_id = "deploy-poller-registered-rates-1"
        litellm.model_cost[deployment_id] = {
            "id": deployment_id,
            "input_cost_per_token_batches": 2e-06,
            "output_cost_per_token_batches": 4e-06,
            "litellm_provider": "bedrock",
            "mode": "chat",
        }

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        mock_job = MagicMock()
        mock_job.id = "job-poller-rates-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "file-output-123"
        mock_response.model_dump_json.return_value = '{"id":"batch-1","status":"completed"}'

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"custom_llm_provider": "bedrock", "aws_region_name": "us-east-1"}
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "bedrock"
        mock_deployment.litellm_params.model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"recordId":"req-1"}'

        decoded_id = f"llm_model_id,{deployment_id};llm_batch_id,batch-456;"

        try:
            with (
                patch(
                    "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                    side_effect=[decoded_id, None],
                ),
                patch(
                    "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                    return_value=deployment_id,
                ),
                patch(
                    "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                    return_value="batch-456",
                ),
                patch(
                    "litellm.files.main.afile_content",
                    new_callable=AsyncMock,
                    return_value=mock_file_content,
                ),
                patch(
                    "litellm.batches.batch_utils._get_file_content_as_dictionary",
                    return_value=[{"recordId": "req-1"}],
                ),
                patch(
                    "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                    new_callable=AsyncMock,
                    return_value=(0.0052, {"prompt_tokens": 1400, "completion_tokens": 600}, ["claude-haiku-4-5"]),
                ) as mock_calculate,
                patch(
                    "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                    return_value=("us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", None, None),
                ),
                patch("litellm.litellm_core_utils.litellm_logging.Logging") as mock_logging_cls,
            ):
                mock_logging_obj = MagicMock()
                mock_logging_obj.async_success_handler = AsyncMock()
                mock_logging_cls.return_value = mock_logging_obj

                await check_batch_cost_instance.check_batch_cost()
        finally:
            litellm.model_cost.pop(deployment_id, None)

        mock_calculate.assert_awaited_once()
        passed_model_info = mock_calculate.await_args.kwargs["model_info"]
        assert passed_model_info is not None, "poller must pass the deployment's registered pricing"
        assert passed_model_info["input_cost_per_token_batches"] == 2e-06
        assert passed_model_info["output_cost_per_token_batches"] == 4e-06

    @pytest.mark.asyncio
    async def test_primary_path_completion_update_includes_batch_processed(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """When batch_processed column IS present, completion update must set it to True.

        This is the symmetric counterpart to test_fallback_completion_update_omits_batch_processed
        and proves the conditional on _has_batch_processed_column governs the update data.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-primary-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "file-output-123"
        mock_response.model_dump_json.return_value = (
            '{"id":"batch-1","status":"completed"}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "openai"
        mock_deployment.litellm_params.model = "gpt-4"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gpt-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-4", "openai", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        ), "Expected update() to be called exactly once for the completed job"
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert (
            update_data["batch_processed"] is True
        ), "update() must include batch_processed=True when column is present"
        assert update_data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_completed_batch_with_no_attributable_owner_still_writes_spend_log(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Regression: a batch created with the master key or a team-less key has
        created_by=None and team_id=None on LiteLLM_ManagedObjectTable (the table
        never stores the raw key hash). CheckBatchCost's synthetic logging_obj for
        such a batch then carries no attributable key/user/team/end-user, and
        before the fix _should_track_cost_callback silently skipped the DB write
        with no error or warning: batch_processed still became True, but no
        LiteLLM_SpendLogs row was ever written.

        Unlike the other tests in this file, this one does NOT mock
        litellm_logging.Logging or async_success_handler -- it runs the real
        logging pipeline through to _ProxyDBLogger, which is the exact gap that
        let the original bug ship undetected.
        """
        import litellm
        from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        mock_job = MagicMock()
        mock_job.id = "job-unattributed-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = None
        mock_job.team_id = None

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])

        # A real LiteLLMBatch (not a bare MagicMock): this test runs the real
        # litellm_logging.Logging pipeline, which type-checks the result via
        # isinstance(..., LiteLLMBatch) before it will compute/attach a cost.
        from litellm.types.utils import LiteLLMBatch

        mock_response = LiteLLMBatch(
            id="batch-1",
            completion_window="24h",
            created_at=1,
            endpoint="/v1/chat/completions",
            input_file_id="file-input-123",
            object="batch",
            status="completed",
            output_file_id="file-output-123",
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(return_value={"api_key": "sk-test"})

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "openai"
        mock_deployment.litellm_params.model = "gpt-4"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        db_logger = _ProxyDBLogger()
        mock_update_database = AsyncMock()

        # Unlike the other tests in this file, this one runs the real
        # litellm_logging.Logging pipeline, which calls
        # _is_base64_encoded_unified_file_id an extra time (checking result.id
        # after it's reset to job.unified_object_id). Key off the argument
        # instead of a fixed-length side_effect list so the exact call count
        # doesn't matter.
        def _fake_is_base64_encoded(file_id):
            return decoded_id if file_id == mock_job.unified_object_id else None

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=_fake_is_base64_encoded,
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gpt-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-4", "openai", None, None),
            ),
            patch.object(litellm, "_async_success_callback", [db_logger]),
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj",
                MagicMock(
                    db_spend_update_writer=MagicMock(update_database=mock_update_database),
                    slack_alerting_instance=MagicMock(customer_spend_alert=AsyncMock()),
                ),
            ),
            patch("litellm.proxy.proxy_server.increment_spend_counters", AsyncMock()),
            patch("litellm.proxy.proxy_server.update_cache", AsyncMock()),
        ):
            await check_batch_cost_instance.check_batch_cost()

        mock_update_database.assert_awaited_once()
        assert mock_update_database.call_args.kwargs["response_cost"] == 0.01
        assert mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1, (
            "the job must still be marked processed once cost tracking succeeds"
        )

    @pytest.mark.asyncio
    async def test_cost_tracking_failure_leaves_job_unprocessed(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """LIT-4008 regression: when fetching a completed batch's results fails
        (e.g. Anthropic rejecting a msgbatch_ id on the Files API), the job must
        NOT be marked complete/batch_processed. Pre-fix the $0 spend row was
        written and batch_processed=True made it permanent; the failure must
        instead leave the row untouched so the next poll retries, without
        aborting the poll cycle.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-anthropic-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "msgbatch_01WA5hdsa2Xx8w4zyPjV1frs"

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test", "custom_llm_provider": "anthropic"}
        )

        decoded_id = "llm_model_id,model-123;llm_batch_id,msgbatch_01WA5hdsa2Xx8w4zyPjV1frs;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="msgbatch_01WA5hdsa2Xx8w4zyPjV1frs",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                side_effect=Exception("File id must have `file_` prefix."),
            ),
        ):
            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 0
        ), "a failed cost tracking attempt must not mark the job processed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal_status", ["failed", "expired", "cancelled"])
    async def test_terminal_status_marks_job_processed(
        self,
        check_batch_cost_instance,
        mock_prisma_client,
        mock_llm_router,
        terminal_status,
    ):
        """When the provider reports a terminal status with nothing to bill
        (failed/cancelled, or expired with no output file), the row must be written back
        with that status and batch_processed=True so it stops being polled forever.
        """
        import base64

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-terminal-1"
        mock_job.unified_object_id = base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        ).decode()
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = terminal_status
        mock_response.output_file_id = None
        mock_response.model_dump_json.return_value = (
            f'{{"id":"batch-1","status":"{terminal_status}"}}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)

        await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        ), f"Expected update() to be called exactly once for a {terminal_status} job"
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert update_data["status"] == terminal_status
        assert (
            update_data["batch_processed"] is True
        ), "terminal-status update() must set batch_processed=True so polling stops"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
    async def test_terminal_status_persists_managed_output_file_ids(
        self,
        check_batch_cost_instance,
        mock_prisma_client,
        mock_llm_router,
        terminal_status,
    ):
        """A cancelled/failed batch with a provider error file (and no output file) must
        be persisted with unified managed file IDs, never raw provider IDs. Raw IDs
        written here leak to every later GET /batches/{id} and GET /batches because the
        terminal row is final (batch_processed=True) and read paths only resolve, never
        mint. (Any terminal status with an output file is billed through the completed
        path instead, covered by test_terminal_status_with_output_file_is_billed.)
        """
        import base64
        import json

        from litellm.types.utils import LiteLLMBatch

        unified_batch_uid = base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        ).decode()
        raw_error_file_id = "file-terminal-err-xyz"
        raw_input_file_id = "file-terminal-in-123"
        unified_input_file_id = base64.urlsafe_b64encode(
            b"litellm_proxy:application/octet-stream;unified_id,in-1;target_model_names,gpt-5-batch"
        ).decode()
        unified_error_file_id = base64.urlsafe_b64encode(
            f"litellm_proxy:application/octet-stream;unified_id,u-2;llm_output_file_id,{raw_error_file_id}".encode()
        ).decode()

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        input_file_row = MagicMock()
        input_file_row.unified_file_id = unified_input_file_id

        def find_managed_file(where):
            if where["flat_model_file_ids"]["has"] == raw_input_file_id:
                return input_file_row
            return None

        mock_prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(
            side_effect=find_managed_file
        )

        mock_job = MagicMock()
        mock_job.id = "job-terminal-mint-1"
        mock_job.unified_object_id = unified_batch_uid
        mock_job.created_by = "user-1"
        mock_job.team_id = "team-1"

        check_batch_cost_instance._has_batch_processed_column = True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        response = LiteLLMBatch(
            id="batch-456",
            completion_window="24h",
            created_at=1,
            endpoint="/v1/chat/completions",
            input_file_id=raw_input_file_id,
            object="batch",
            status=terminal_status,
            output_file_id=None,
            error_file_id=raw_error_file_id,
        )
        mock_llm_router.aretrieve_batch = AsyncMock(return_value=response)

        mock_hook = MagicMock()
        mock_hook.get_unified_output_file_id.side_effect = [unified_error_file_id]
        mock_hook.store_unified_file_id = AsyncMock()
        check_batch_cost_instance.proxy_logging_obj.get_proxy_hook.return_value = (
            mock_hook
        )

        await check_batch_cost_instance.check_batch_cost()

        mock_hook.get_unified_output_file_id.assert_called_once_with(
            output_file_id=raw_error_file_id,
            model_id="model-123",
            model_name="gpt-5-batch",
        )
        stored = {
            next(iter(c.kwargs["model_mappings"].values())): c.kwargs["file_id"]
            for c in mock_hook.store_unified_file_id.call_args_list
        }
        assert stored == {raw_error_file_id: unified_error_file_id}
        for store_call in mock_hook.store_unified_file_id.call_args_list:
            assert store_call.kwargs["user_api_key_dict"].user_id == "user-1"
            assert store_call.kwargs["user_api_key_dict"].team_id == "team-1"

        assert mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        update_call = mock_prisma_client.db.litellm_managedobjecttable.update.call_args
        assert update_call.kwargs["where"] == {"id": "job-terminal-mint-1"}
        update_data = update_call.kwargs["data"]
        assert update_data["status"] == terminal_status
        assert update_data["batch_processed"] is True
        persisted = json.loads(update_data["file_object"])
        assert persisted["id"] == unified_batch_uid
        assert persisted["input_file_id"] == unified_input_file_id
        assert persisted["output_file_id"] is None
        assert persisted["error_file_id"] == unified_error_file_id
        assert raw_error_file_id not in update_data["file_object"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("completed_status", ["completed", "complete"])
    async def test_completed_without_output_file_marked_processed_without_billing(
        self,
        check_batch_cost_instance,
        mock_prisma_client,
        mock_llm_router,
        completed_status,
    ):
        """#35354 regression: a terminal completed batch whose request lines all failed
        reaches `completed` with output_file_id=None (only an error_file_id).

        Pre-fix it matched neither the completed-with-output branch nor the
        failed/expired/cancelled branch, so batch_processed stayed False and the row
        was re-selected on every poll cycle forever. It must now be marked terminal
        exactly once, without being billed: request_counts.completed == 0 proves the
        missing output file means nothing to bill rather than a lagging output id
        (#37713 keeps the lagging case eligible for the next cycle).
        """
        import base64
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-completed-no-output-1"
        mock_job.unified_object_id = base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        ).decode()
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = completed_status
        mock_response.output_file_id = None
        mock_response.error_file_id = "file-error-123"
        mock_response.request_counts = MagicMock(completed=0, failed=3, total=3)
        mock_response.model_dump_json.return_value = (
            f'{{"id":"batch-1","status":"{completed_status}"}}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        # Billing reads credentials off the router; if it is touched we billed a batch
        # that has no output, which is the behaviour this test guards against.
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        with patch(
            "litellm.files.main.afile_content",
            new_callable=AsyncMock,
        ) as mock_afile_content:
            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        ), "a completed batch with no output file must be marked processed exactly once"
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert update_data["status"] == completed_status
        assert (
            update_data["batch_processed"] is True
        ), "completed-without-output update() must set batch_processed=True so polling stops"
        assert (
            mock_afile_content.await_count == 0
        ), "a batch with no output file must not be billed"
        assert (
            mock_llm_router.get_deployment_credentials_with_provider.call_count == 0
        ), "a batch with no output file must not enter the cost-tracking path"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_counts",
        [MagicMock(completed=7, failed=0, total=7), None],
        ids=["lagging_output_id", "unknown_counts"],
    )
    async def test_completed_with_lagging_output_file_left_for_next_cycle(
        self,
        check_batch_cost_instance,
        mock_prisma_client,
        mock_llm_router,
        request_counts,
    ):
        """#37713 regression: a batch can report completed while its output_file_id is
        still lagging behind at the provider. Retiring it in that window (or when the
        request counts cannot prove there is nothing to bill) permanently loses the
        spend record, so the poller must leave the row untouched and revisit it on the
        next cycle once the output id has appeared.
        """
        import base64
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-completed-lagging-output-1"
        mock_job.unified_object_id = base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        ).decode()
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = None
        mock_response.error_file_id = None
        mock_response.request_counts = request_counts

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        with patch(
            "litellm.files.main.afile_content",
            new_callable=AsyncMock,
        ) as mock_afile_content:
            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 0
        ), "a completed batch whose output id is still lagging must stay eligible for the next poll"
        assert (
            mock_afile_content.await_count == 0
        ), "a batch with no output file must not be billed"

    @pytest.mark.asyncio
    async def test_non_terminal_status_left_unprocessed(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A batch still validating/in_progress must NOT be treated as terminal: no DB
        write, so it keeps being polled until it actually reaches a terminal status.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()

        mock_job = MagicMock()
        mock_job.id = "job-in-progress-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = "in_progress"
        mock_response.output_file_id = None

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
        ):
            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 0
        ), "a non-terminal batch must not be written back (would stop polling prematurely)"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal_status", ["expired", "cancelled", "failed"])
    async def test_terminal_status_with_output_file_is_billed(
        self,
        check_batch_cost_instance,
        mock_prisma_client,
        mock_llm_router,
        terminal_status,
    ):
        """A terminal (expired/cancelled/failed) batch that still produced an output file
        served real request lines, so it must be billed (cost tracked) and then marked
        processed, not silently marked terminal without billing.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-terminal-with-output-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = terminal_status
        mock_response.output_file_id = "file-output-123"
        mock_response.model_dump_json.return_value = (
            f'{{"id":"batch-1","status":"{terminal_status}"}}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "openai"
        mock_deployment.litellm_params.model = "gpt-4"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                side_effect=[decoded_id, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ) as mock_afile_content,
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gpt-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-4", "openai", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await check_batch_cost_instance.check_batch_cost()

        assert (
            mock_afile_content.await_count == 1
        ), f"{terminal_status} batch with an output file must fetch results and be billed"
        mock_logging_obj.async_success_handler.assert_awaited_once()
        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        )
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert update_data["batch_processed"] is True
        assert (
            update_data["status"] == terminal_status
        ), f"billed {terminal_status} batch must keep its real terminal status in the DB"

    @pytest.mark.asyncio
    async def test_terminal_batch_with_missing_output_file_is_retired_unbilled(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A terminal batch whose advertised output file 404s at the provider has
        nothing to fetch on this or any later poll (Vertex AI advertises an output
        path for every batch, even ones that never wrote it), so the job must be
        retired as terminal on the first cycle instead of retrying until the
        staleness sweep gives up on it.
        """
        import base64
        from unittest.mock import patch

        from litellm.exceptions import NotFoundError

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-output-gone-1"
        mock_job.unified_object_id = base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        ).decode()
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        missing_output_file_id = "gs://batch-out/job-1/predictions.jsonl"
        mock_response = MagicMock()
        mock_response.status = "failed"
        mock_response.output_file_id = missing_output_file_id
        mock_response.error_file_id = None
        mock_response.model_dump_json.return_value = (
            '{"id":"batch-1","status":"failed"}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        with (
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                side_effect=NotFoundError(
                    message=f"404: output file {missing_output_file_id} does not exist",
                    model="gemini-2.5-pro",
                    llm_provider="vertex_ai",
                ),
            ) as mock_afile_content,
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
            ) as mock_calculate,
        ):
            await check_batch_cost_instance.check_batch_cost()

        assert mock_afile_content.await_count == 1
        mock_calculate.assert_not_awaited()
        assert (
            mock_prisma_client.db.litellm_managedobjecttable.update.call_count == 1
        ), "a terminal batch with a 404ing output file must be retired, not retried forever"
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[
            1
        ]["data"]
        assert update_data["status"] == "failed"
        assert update_data["batch_processed"] is True

    @pytest.mark.asyncio
    async def test_raw_output_file_id_converted_to_managed_id(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """CheckBatchCost must convert a raw provider output_file_id to a managed base64 ID.

        Without this, GET /batches/{id} returns a raw file ID that cannot be routed
        through the proxy, causing API_KEY errors when clients call GET /files/{id}/content.
        """
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-raw-file-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"
        mock_job.team_id = None

        check_batch_cost_instance._has_batch_processed_column = True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        raw_output_file_id = "file-batch-output-abc123"
        raw_error_file_id = "file-batch-error-xyz456"
        fake_managed_output_id = "bGl0ZWxsbV9wcm94eTo6b3V0cHV0"
        fake_managed_error_id = "bGl0ZWxsbV9wcm94eTo6ZXJyb3I="

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = raw_output_file_id
        mock_response.error_file_id = raw_error_file_id
        mock_response.model_dump_json.return_value = (
            '{"id":"batch-1","status":"completed"}'
        )

        mock_llm_router.aretrieve_batch = AsyncMock(return_value=mock_response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )

        mock_deployment = MagicMock()
        mock_deployment.litellm_params.custom_llm_provider = "azure"
        mock_deployment.litellm_params.model = "azure/gpt-5-mini"
        mock_deployment.model_name = "gpt-5-batch"
        mock_deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=mock_deployment)

        mock_hook = MagicMock()
        mock_hook.get_unified_output_file_id.side_effect = [
            fake_managed_output_id,
            fake_managed_error_id,
        ]
        mock_hook.store_unified_file_id = AsyncMock()
        check_batch_cost_instance.proxy_logging_obj.get_proxy_hook.return_value = (
            mock_hook
        )

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'
        decoded_id = "llm_model_id,model-123;llm_batch_id,batch-456;"

        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                # call 1: job unified_object_id decode, call 2: existing raw check for output_file_id,
                # call 3: fix guard for output_file_id, call 4: fix guard for error_file_id
                side_effect=[decoded_id, None, None, None],
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gpt-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-5-mini", "azure", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await check_batch_cost_instance.check_batch_cost()

        assert mock_hook.get_unified_output_file_id.call_count == 2
        mock_hook.get_unified_output_file_id.assert_any_call(
            output_file_id=raw_output_file_id,
            model_id="model-123",
            model_name="gpt-5-batch",
        )
        mock_hook.get_unified_output_file_id.assert_any_call(
            output_file_id=raw_error_file_id,
            model_id="model-123",
            model_name="gpt-5-batch",
        )
        assert mock_hook.store_unified_file_id.await_count == 2
        # {raw_file_id: managed_file_id} for each store call
        stored = {
            next(iter(c[1]["model_mappings"].values())): c[1]["file_id"]
            for c in mock_hook.store_unified_file_id.call_args_list
        }
        assert stored == {
            raw_output_file_id: fake_managed_output_id,
            raw_error_file_id: fake_managed_error_id,
        }
        assert mock_response.output_file_id == fake_managed_output_id
        assert mock_response.error_file_id == fake_managed_error_id


class TestUnmanagedVertexRouting:
    """Routing of unmanaged Vertex batches whose unified_object_id is a raw provider job id."""

    def _instance(self, track_unmanaged, router):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )

        return CheckBatchCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=MagicMock(),
            llm_router=router,
            track_unmanaged_batch_cost=track_unmanaged,
        )

    def _job(self, file_object=None):
        job = MagicMock()
        job.unified_object_id = "8823717160934178816"
        job.file_object = (
            file_object if file_object is not None else _unmanaged_vertex_file_object()
        )
        return job

    def test_flag_off_skips_unmanaged_id_unchanged(self):
        """Default (flag off): a raw numeric unified_object_id is skipped exactly as before;
        no model derivation or router lookup happens."""
        router = MagicMock()
        instance = self._instance(track_unmanaged=False, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with("invalid_unified_id")
        router.resolve_model_name_from_model_id.assert_not_called()
        router.get_model_ids.assert_not_called()

    def _vertex_deployment(self):
        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "vertex_ai"
        deployment.litellm_params.model = "vertex_ai/gemini-2.5-flash"
        return deployment

    def test_flag_on_routes_to_vertex_deployment(self):
        """Flag on: derive the bare model from the gs:// path, resolve it to a deployment id,
        and use the raw unified_object_id as the provider batch id."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "gemini-2.5-flash"
        router.get_model_ids.return_value = ["deploy-1"]
        router.get_deployment = MagicMock(return_value=self._vertex_deployment())
        instance = self._instance(track_unmanaged=True, router=router)

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), MagicMock())

        assert result == ("deploy-1", "8823717160934178816")
        # bare model name (trailing GCS segment), not the full publishers/.. path
        router.resolve_model_name_from_model_id.assert_called_once_with(
            "gemini-2.5-flash"
        )
        router.get_model_ids.assert_called_once_with(model_name="gemini-2.5-flash")

    def test_flag_on_skips_non_vertex_deployment_sharing_model_group(self):
        """Flag on, but the only deployment for the model group is a non-vertex_ai
        provider: must not be selected, even though the model group name matches."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "gemini-2.5-flash"
        router.get_model_ids.return_value = ["deploy-openai"]
        non_vertex_deployment = MagicMock()
        non_vertex_deployment.litellm_params.custom_llm_provider = "openai"
        non_vertex_deployment.litellm_params.model = "gpt-4o"
        router.get_deployment = MagicMock(return_value=non_vertex_deployment)
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with(
            "unmanaged_no_matching_deployment"
        )

    def test_flag_on_uses_later_vertex_deployment_with_matching_suffix(self):
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "azure-gemini"
        router.get_model_ids.return_value = ["deploy-azure"]
        non_vertex_deployment = MagicMock()
        non_vertex_deployment.litellm_params.custom_llm_provider = "azure"
        non_vertex_deployment.litellm_params.model = "azure/gemini-2.5-flash"
        router.get_deployment = MagicMock(return_value=non_vertex_deployment)
        router.get_model_list.return_value = [
            {
                "model_name": "azure-gemini",
                "litellm_params": {
                    "model": "azure/gemini-2.5-flash",
                    "custom_llm_provider": "azure",
                },
                "model_info": {"id": "deploy-azure"},
            },
            {
                "model_name": "vertex-gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-flash",
                    "custom_llm_provider": "vertex_ai",
                },
                "model_info": {"id": "deploy-vertex"},
            },
        ]
        instance = self._instance(track_unmanaged=True, router=router)

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), MagicMock())

        assert result == ("deploy-vertex", "8823717160934178816")
        router.get_model_ids.assert_called_once_with(model_name="azure-gemini")

    def test_flag_on_no_matching_deployment_records_metric(self):
        """Flag on but no vertex_ai deployment for the model: skip with a distinct metric."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = None
        router.get_model_ids.return_value = []
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with(
            "unmanaged_no_matching_deployment"
        )

    def test_flag_on_non_gcs_input_is_not_unmanaged_vertex(self):
        """Flag on, but input_file_id is not a gs:// publishers path: treat as unroutable,
        do not attempt model derivation."""
        router = MagicMock()
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()
        job = self._job(
            file_object=_unmanaged_vertex_file_object(input_file_id="file-abc-123")
        )

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(job, prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with("invalid_unified_id")
        router.resolve_model_name_from_model_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_to_end_costs_unmanaged_batch(self):
        """Flag on, completed unmanaged batch: the poller polls Vertex with the raw job id,
        computes cost, and marks batch_processed=True. Fails before this change (the row is
        skipped at the unified-id gate)."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "gemini-2.5-flash"
        router.get_model_ids.return_value = ["deploy-1"]

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "gs://bucket/out/predictions.jsonl"
        mock_response.error_file_id = None
        mock_response.completed_at = None
        mock_response.created_at = None
        mock_response.model_dump_json.return_value = (
            '{"id":"8823717160934178816","status":"completed"}'
        )
        router.aretrieve_batch = AsyncMock(return_value=mock_response)
        router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"vertex_project": "p", "vertex_location": "us-central1"}
        )

        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "vertex_ai"
        deployment.litellm_params.model = "vertex_ai/gemini-2.5-flash"
        deployment.model_name = "gemini-2.5-flash"
        deployment.model_info.model_dump.return_value = {}
        router.get_deployment = MagicMock(return_value=deployment)

        instance = self._instance(track_unmanaged=True, router=router)
        instance.proxy_logging_obj.get_proxy_hook.return_value = None
        instance._has_batch_processed_column = True

        prisma = instance.prisma_client
        prisma.db = MagicMock()
        prisma.db.litellm_managedobjecttable = MagicMock()
        prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        prisma.db.litellm_managedobjecttable.update = AsyncMock()
        prisma.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[self._job()]
        )
        prisma.db.litellm_usertable = MagicMock()
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        with (
            patch(_IS_B64, side_effect=[False, None]),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.01,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["gemini-2.5-flash"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gemini-2.5-flash", "vertex_ai", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await instance.check_batch_cost()

        router.aretrieve_batch.assert_awaited_once()
        assert router.aretrieve_batch.call_args[1]["model"] == "deploy-1"
        assert router.aretrieve_batch.call_args[1]["batch_id"] == "8823717160934178816"

        mock_logging_obj.async_success_handler.assert_awaited_once()
        assert mock_logging_obj.async_success_handler.call_args[1]["batch_cost"] == 0.01

        assert prisma.db.litellm_managedobjecttable.update.call_count == 1
        update_data = prisma.db.litellm_managedobjecttable.update.call_args[1]["data"]
        assert update_data["batch_processed"] is True
        assert update_data["status"] == "complete"


class TestUnmanagedBedrockRouting:
    """Routing of unmanaged Bedrock batches whose unified_object_id is a raw model-invocation-job ARN."""

    _ARN = "arn:aws:bedrock:us-east-1:298249409318:model-invocation-job/1ofb47x17jua"

    def _instance(self, track_unmanaged, router):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )

        return CheckBatchCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=MagicMock(),
            llm_router=router,
            track_unmanaged_batch_cost=track_unmanaged,
        )

    def _job(self, file_object=None):
        job = MagicMock()
        job.unified_object_id = self._ARN
        job.file_object = (
            file_object if file_object is not None else _unmanaged_bedrock_file_object()
        )
        return job

    def _bedrock_deployment(self):
        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "bedrock"
        deployment.litellm_params.model = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
        return deployment

    def test_flag_off_skips_arn_unified_id_unchanged(self):
        """Default (flag off): a raw ARN unified_object_id is skipped exactly as before; no
        model derivation or router lookup happens."""
        router = MagicMock()
        instance = self._instance(track_unmanaged=False, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with("invalid_unified_id")
        router.resolve_model_name_from_model_id.assert_not_called()
        router.get_model_ids.assert_not_called()

    def test_flag_on_routes_to_bedrock_deployment(self):
        """Flag on: derive the bare model from the s3:// object key (":" restored to "-" is
        matched fuzzily), resolve it to a deployment id, and use the raw ARN as the batch id."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "claude-sonnet-4"
        router.get_model_ids.return_value = ["deploy-1"]
        router.get_deployment = MagicMock(return_value=self._bedrock_deployment())
        instance = self._instance(track_unmanaged=True, router=router)

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), MagicMock())

        assert result == ("deploy-1", self._ARN)

    def test_flag_on_skips_non_bedrock_deployment_sharing_model_group(self):
        """Flag on, but the only deployment for the model group is a non-bedrock provider:
        must not be selected, even though the model group name matches."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "claude-sonnet-4"
        router.get_model_ids.return_value = ["deploy-anthropic"]
        non_bedrock_deployment = MagicMock()
        non_bedrock_deployment.litellm_params.custom_llm_provider = "anthropic"
        non_bedrock_deployment.litellm_params.model = "claude-sonnet-4-20250514"
        router.get_deployment = MagicMock(return_value=non_bedrock_deployment)
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with(
            "unmanaged_no_matching_deployment"
        )

    def test_flag_on_matches_deployment_despite_colon_dash_mismatch(self):
        """The S3 object key has ':' replaced with '-' (e.g. 'v1-0'), but the configured
        deployment's actual bedrock model id uses ':' (e.g. 'v1:0'). Routing must still match."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = None
        router.get_model_ids.return_value = []
        router.get_model_list.return_value = [
            {
                "model_name": "claude-sonnet-4",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "custom_llm_provider": "bedrock",
                },
                "model_info": {"id": "deploy-bedrock"},
            }
        ]
        instance = self._instance(track_unmanaged=True, router=router)

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), MagicMock())

        assert result == ("deploy-bedrock", self._ARN)

    def test_flag_on_no_matching_deployment_records_metric(self):
        """Flag on but no bedrock deployment for the model: skip with a distinct metric."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = None
        router.get_model_ids.return_value = []
        router.get_model_list.return_value = []
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(self._job(), prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with(
            "unmanaged_no_matching_deployment"
        )

    def test_flag_on_non_s3_input_is_not_unmanaged_bedrock(self):
        """Flag on, but input_file_id is not a litellm-bedrock-files- s3:// key: treat as
        unroutable, do not attempt model derivation."""
        router = MagicMock()
        instance = self._instance(track_unmanaged=True, router=router)
        prom = MagicMock()
        job = self._job(
            file_object=_unmanaged_bedrock_file_object(input_file_id="file-abc-123")
        )

        with patch(_IS_B64, return_value=False):
            result = instance._resolve_job_routing(job, prom)

        assert result is None
        prom.record_check_batch_cost_error.assert_called_once_with("invalid_unified_id")
        router.resolve_model_name_from_model_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_to_end_costs_unmanaged_batch(self):
        """Flag on, completed unmanaged batch: the poller polls Bedrock with the raw ARN,
        computes cost, and marks batch_processed=True."""
        router = MagicMock()
        router.resolve_model_name_from_model_id.return_value = "claude-sonnet-4"
        router.get_model_ids.return_value = ["deploy-1"]

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_file_id = "s3://bucket/out/predictions.jsonl"
        mock_response.error_file_id = None
        mock_response.completed_at = None
        mock_response.created_at = None
        mock_response.model_dump_json.return_value = (
            f'{{"id":"{self._ARN}","status":"completed"}}'
        )
        router.aretrieve_batch = AsyncMock(return_value=mock_response)
        router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"aws_region_name": "us-east-1"}
        )

        deployment = self._bedrock_deployment()
        deployment.model_name = "claude-sonnet-4"
        deployment.model_info.model_dump.return_value = {}
        router.get_deployment = MagicMock(return_value=deployment)

        instance = self._instance(track_unmanaged=True, router=router)
        instance.proxy_logging_obj.get_proxy_hook.return_value = None
        instance._has_batch_processed_column = True

        prisma = instance.prisma_client
        prisma.db = MagicMock()
        prisma.db.litellm_managedobjecttable = MagicMock()
        prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        prisma.db.litellm_managedobjecttable.update = AsyncMock()
        prisma.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[self._job()]
        )
        prisma.db.litellm_usertable = MagicMock()
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        mock_file_content = MagicMock()
        mock_file_content.content = b'{"id":"req-1"}'

        with (
            patch(_IS_B64, side_effect=[False, None]),
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=mock_file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(
                    0.02,
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    ["claude-sonnet-4"],
                ),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("claude-sonnet-4", "bedrock", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.litellm_logging.Logging"
            ) as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj

            await instance.check_batch_cost()

        router.aretrieve_batch.assert_awaited_once()
        assert router.aretrieve_batch.call_args[1]["model"] == "deploy-1"
        assert router.aretrieve_batch.call_args[1]["batch_id"] == self._ARN

        mock_logging_obj.async_success_handler.assert_awaited_once()
        assert mock_logging_obj.async_success_handler.call_args[1]["batch_cost"] == 0.02

        assert prisma.db.litellm_managedobjecttable.update.call_count == 1
        update_data = prisma.db.litellm_managedobjecttable.update.call_args[1]["data"]
        assert update_data["batch_processed"] is True
        assert update_data["status"] == "complete"


class TestUnmanagedBatchCostFlagIsGeneralized:
    """The single track_unmanaged_batch_cost flag must cover both Vertex and Bedrock, not
    just the provider it was originally added for."""

    def test_one_flag_routes_both_vertex_and_bedrock_jobs(self):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )

        router = MagicMock()
        router.resolve_model_name_from_model_id.side_effect = [
            "gemini-2.5-flash",
            "claude-sonnet-4",
        ]
        router.get_model_ids.side_effect = [["deploy-vertex"], ["deploy-bedrock"]]

        def _get_deployment(model_id):
            if model_id == "deploy-vertex":
                deployment = MagicMock()
                deployment.litellm_params.custom_llm_provider = "vertex_ai"
                deployment.litellm_params.model = "vertex_ai/gemini-2.5-flash"
                return deployment
            deployment = MagicMock()
            deployment.litellm_params.custom_llm_provider = "bedrock"
            deployment.litellm_params.model = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
            return deployment

        router.get_deployment = MagicMock(side_effect=_get_deployment)

        instance = CheckBatchCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=MagicMock(),
            llm_router=router,
            track_unmanaged_batch_cost=True,
        )

        vertex_job = MagicMock()
        vertex_job.unified_object_id = "8823717160934178816"
        vertex_job.file_object = _unmanaged_vertex_file_object()

        bedrock_job = MagicMock()
        bedrock_job.unified_object_id = TestUnmanagedBedrockRouting._ARN
        bedrock_job.file_object = _unmanaged_bedrock_file_object()

        with patch(_IS_B64, return_value=False):
            vertex_result = instance._resolve_job_routing(vertex_job, MagicMock())
            bedrock_result = instance._resolve_job_routing(bedrock_job, MagicMock())

        assert vertex_result == ("deploy-vertex", "8823717160934178816")
        assert bedrock_result == ("deploy-bedrock", TestUnmanagedBedrockRouting._ARN)


class TestManagedOutputFileIdEncodesPublicModelGroup:
    """LIT-4964 regression: the unified output file id created by the background poller must
    encode the public model group as ``target_model_names``, not the provider model.

    Key model-access checks resolve a managed file id back to a model via ``target_model_names``,
    so encoding the provider model (e.g. ``gpt-5.5``) makes
    ``GET /v1/files/{output_file_id}/content`` fail for every key.
    """

    _PUBLIC_MODEL_GROUP = "gpt-5-batch"
    _RAW_OUTPUT_FILE_ID = "file-batch-output-abc123"

    @staticmethod
    def _managed_input_file_id(model_group: str) -> str:
        import base64

        unified_id = (
            "litellm_proxy:application/octet-stream;unified_id,c4843482-b176-4901-8292-7523fd0f2c6e;"
            f"target_model_names,{model_group}"
        )
        return base64.urlsafe_b64encode(unified_id.encode()).decode().rstrip("=")

    def _job(self, input_file_id: str) -> MagicMock:
        from litellm.types.utils import LiteLLMBatch

        job = MagicMock()
        job.id = "job-lit-4964"
        job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        job.created_by = "user-1"
        job.team_id = None
        job.file_object = LiteLLMBatch(
            id="batch-456",
            completion_window="24h",
            created_at=1,
            endpoint="/v1/chat/completions",
            input_file_id=input_file_id,
            object="batch",
            status="completed",
        ).model_dump_json()
        return job

    async def _run(self, job: MagicMock) -> str:
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )
        from litellm.types.utils import LiteLLMBatch
        from enterprise.litellm_enterprise.proxy.hooks.managed_files import (
            _PROXY_LiteLLMManagedFiles,
        )

        router = MagicMock()
        router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )
        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "azure"
        deployment.litellm_params.model = "azure/gpt-5.5"
        deployment.model_name = self._PUBLIC_MODEL_GROUP
        deployment.model_info.model_dump.return_value = {}
        router.get_deployment = MagicMock(return_value=deployment)

        hook = MagicMock()
        hook.get_unified_output_file_id = (
            lambda output_file_id, model_id, model_name: _PROXY_LiteLLMManagedFiles.get_unified_output_file_id(
                None, output_file_id=output_file_id, model_id=model_id, model_name=model_name
            )
        )
        hook.store_unified_file_id = AsyncMock()
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.get_proxy_hook.return_value = hook

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        instance = CheckBatchCost(
            proxy_logging_obj=proxy_logging_obj,
            prisma_client=prisma_client,
            llm_router=router,
        )

        response = LiteLLMBatch(
            id="batch-456",
            completion_window="24h",
            created_at=1,
            endpoint="/v1/chat/completions",
            input_file_id=job.file_object,
            object="batch",
            status="completed",
        )
        response.output_file_id = self._RAW_OUTPUT_FILE_ID

        file_content = MagicMock()
        file_content.content = b'{"id":"req-1"}'

        with (
            patch(
                "litellm.files.main.afile_content",
                new_callable=AsyncMock,
                return_value=file_content,
            ),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(0.01, {"prompt_tokens": 10}, ["gpt-5.5"]),
            ),
            patch("litellm.litellm_core_utils.litellm_logging.Logging") as logging_cls,
        ):
            logging_obj = MagicMock()
            logging_obj.async_success_handler = AsyncMock()
            logging_cls.return_value = logging_obj

            await instance._track_completed_batch_cost(
                job=job,
                response=response,
                model_id="model-123",
                batch_id="batch-456",
                prom_logger=None,
            )

        return response.output_file_id

    @pytest.mark.asyncio
    async def test_target_model_names_comes_from_input_file_not_provider_model(self):
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            get_models_from_unified_file_id,
        )

        output_file_id = await self._run(
            self._job(self._managed_input_file_id(self._PUBLIC_MODEL_GROUP))
        )

        decoded = _is_base64_encoded_unified_file_id(output_file_id)
        assert get_models_from_unified_file_id(decoded) == [self._PUBLIC_MODEL_GROUP]
        assert "gpt-5.5" not in decoded

    @pytest.mark.asyncio
    async def test_key_scoped_to_model_group_can_read_the_output_file(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.auth_checks import can_key_call_model
        from litellm.proxy.auth.auth_utils import (
            _extract_models_from_managed_resource_id,
        )

        output_file_id = await self._run(
            self._job(self._managed_input_file_id(self._PUBLIC_MODEL_GROUP))
        )

        models = _extract_models_from_managed_resource_id(output_file_id, "file_id", None)
        assert models == [self._PUBLIC_MODEL_GROUP]
        assert (
            await can_key_call_model(
                model=models[0],
                llm_model_list=None,
                valid_token=UserAPIKeyAuth(
                    api_key="sk-test", models=[self._PUBLIC_MODEL_GROUP]
                ),
                llm_router=None,
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_deployment_model_group_without_managed_input_file(self):
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            get_models_from_unified_file_id,
        )

        output_file_id = await self._run(self._job("file-raw-provider-input"))

        decoded = _is_base64_encoded_unified_file_id(output_file_id)
        assert get_models_from_unified_file_id(decoded) == [self._PUBLIC_MODEL_GROUP]
class TestBatchCostAttribution:
    """CheckBatchCost rebuilds the creator's spend metadata from the managed-object row so
    the batch-cost log is attributed like a non-batch request."""

    def _instance(self, key_row=None, team_row=None, user_row=None):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

        prisma = MagicMock()
        prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=key_row)
        prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)
        return CheckBatchCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=prisma,
            llm_router=MagicMock(),
        )

    def _job(self, **overrides):
        from types import SimpleNamespace

        fields = {
            "created_by": "alice",
            "team_id": "team-alpha",
            "api_key": "hash-alice",
            "request_tags": ["env:prod"],
        }
        fields.update(overrides)
        return SimpleNamespace(unified_object_id="uoi", **fields)

    @pytest.mark.asyncio
    async def test_metadata_carries_key_team_and_tags(self):
        """The spend row names the creating key, its team, both aliases, and the tags."""
        from types import SimpleNamespace

        instance = self._instance(
            key_row=SimpleNamespace(key_alias="prod-key"),
            team_row=SimpleNamespace(team_alias="Team Alpha"),
            user_row=SimpleNamespace(user_email="alice@example.com", user_alias=None),
        )

        metadata = await instance._build_creator_attribution_metadata(self._job(), "batch-1")

        assert metadata["user_api_key"] == "hash-alice"
        assert metadata["user_api_key_user_id"] == "alice"
        assert metadata["user_api_key_team_id"] == "team-alpha"
        assert metadata["user_api_key_alias"] == "prod-key"
        assert metadata["user_api_key_team_alias"] == "Team Alpha"
        assert metadata["tags"] == ["env:prod"]

    @pytest.mark.asyncio
    async def test_metadata_tolerates_legacy_row_without_columns(self):
        """Rows created before the columns existed carry only created_by/team_id and must
        still produce an attributed row rather than raising."""
        instance = self._instance()
        job = self._job(api_key=None, request_tags=None)

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["user_api_key"] is None
        assert metadata["user_api_key_user_id"] == "alice"
        assert metadata["user_api_key_team_id"] == "team-alpha"
        assert "tags" not in metadata

    @pytest.mark.asyncio
    async def test_metadata_keeps_key_when_team_key_has_no_user(self):
        """A team-scoped key carries no user id. The user lookup is skipped (prisma rejects
        a None user_id) and the key hash still drives key-level attribution."""
        from types import SimpleNamespace

        instance = self._instance(key_row=SimpleNamespace(key_alias="svc-key"))
        job = self._job(created_by=None)

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["user_api_key"] == "hash-alice"
        assert metadata["user_api_key_user_id"] is None
        assert metadata["user_api_key_alias"] == "svc-key"
        instance.prisma_client.db.litellm_usertable.find_unique.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_drops_non_string_tags(self):
        """Non-string tags are dropped so a malformed stored value cannot slip past the
        tag-budget checks that consume this metadata."""
        instance = self._instance()
        job = self._job(request_tags=["env:prod", 7, None, "team:ml"])

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["tags"] == ["env:prod", "team:ml"]

    @pytest.mark.asyncio
    async def test_key_alias_lookup_failure_does_not_break_attribution(self):
        """An alias lookup failure must not lose the spend row; the key hash and team still
        attribute it."""
        instance = self._instance()
        instance.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
            side_effect=Exception("db down")
        )

        metadata = await instance._build_creator_attribution_metadata(self._job(), "batch-1")

        assert metadata["user_api_key"] == "hash-alice"
        assert metadata.get("user_api_key_alias") is None

    @pytest.mark.asyncio
    async def test_unnamed_key_keeps_the_creating_user_alias(self):
        """Regression: a key generated without key_alias resolves to no alias, and the
        overwrite must not null out the creating user's alias that _get_user_info supplied.
        Most keys carry no alias, so this is the common batch, not an edge case."""
        from types import SimpleNamespace

        instance = self._instance(
            key_row=SimpleNamespace(key_alias=None),
            user_row=SimpleNamespace(user_email="alice@example.com", user_alias="Alice Chen"),
        )

        metadata = await instance._build_creator_attribution_metadata(self._job(), "batch-1")

        assert metadata["user_api_key_alias"] == "Alice Chen"
        assert metadata["user_api_key"] == "hash-alice"

    @pytest.mark.asyncio
    async def test_rotated_key_keeps_the_creating_user_alias(self):
        """Batches outlive keys. When the creating key has been rotated or deleted the
        lookup returns no row, and the spend log keeps a resolvable name instead of null."""
        from types import SimpleNamespace

        instance = self._instance(
            key_row=None,
            user_row=SimpleNamespace(user_email="alice@example.com", user_alias="Alice Chen"),
        )

        metadata = await instance._build_creator_attribution_metadata(self._job(), "batch-1")

        assert metadata["user_api_key_alias"] == "Alice Chen"

    @pytest.mark.asyncio
    async def test_named_key_still_owns_the_alias(self):
        """The fallback must not weaken the intended precedence: a key that has its own
        alias still overrides the creating user's."""
        from types import SimpleNamespace

        instance = self._instance(
            key_row=SimpleNamespace(key_alias="prod-key"),
            user_row=SimpleNamespace(user_email="alice@example.com", user_alias="Alice Chen"),
        )

        metadata = await instance._build_creator_attribution_metadata(self._job(), "batch-1")

        assert metadata["user_api_key_alias"] == "prod-key"


class TestPollPageStarvation:
    """LIT-5462 regression: a row that can never be costed used to keep its slot in the
    MAX_OBJECTS_PER_POLL_CYCLE page forever, so once enough of them accumulated no newer
    batch was ever polled or costed."""

    def _instance(self, prisma, llm_router):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.get_proxy_hook.return_value = None
        return CheckBatchCost(
            proxy_logging_obj=proxy_logging_obj,
            prisma_client=prisma,
            llm_router=llm_router,
        )

    def _prisma(self, jobs):
        prisma = MagicMock()
        prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=0)
        prisma.db.litellm_managedobjecttable.update = AsyncMock()
        prisma.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=jobs)
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
        return prisma

    def _job(self, job_id, unified_object_id):
        job = MagicMock()
        job.id = job_id
        job.unified_object_id = unified_object_id
        job.created_by = "user-1"
        return job

    @staticmethod
    def _encode(unified_id: str) -> str:
        import base64

        return base64.urlsafe_b64encode(unified_id.encode()).decode().rstrip("=")

    @pytest.mark.asyncio
    async def test_unified_id_without_model_id_is_retired(self):
        """A unified id that decodes but carries no model_id is unroutable no matter what
        the config says, so it must leave the poll page instead of being retried forever."""
        prisma = self._prisma(
            [self._job("job-no-model", self._encode("litellm_proxy;llm_batch_id:poison-no-model"))]
        )
        llm_router = MagicMock()
        llm_router.aretrieve_batch = AsyncMock()

        await self._instance(prisma, llm_router).check_batch_cost()

        llm_router.aretrieve_batch.assert_not_awaited()
        prisma.db.litellm_managedobjecttable.update.assert_awaited_once()
        call = prisma.db.litellm_managedobjecttable.update.call_args[1]
        assert call["where"] == {"id": "job-no-model"}
        assert call["data"] == {"batch_processed": True}

    @pytest.mark.asyncio
    async def test_provider_404_retires_job(self):
        """The provider dropping its record of the batch is permanent: no later retrieve
        can succeed, so the row must stop occupying a slot."""
        import litellm

        prisma = self._prisma(
            [
                self._job(
                    "job-gone",
                    self._encode("litellm_proxy;model_id:model-123;llm_batch_id:batch_deadbeef"),
                )
            ]
        )
        llm_router = MagicMock()
        llm_router.aretrieve_batch = AsyncMock(
            side_effect=litellm.NotFoundError(
                message="No batch found with id 'batch_deadbeef'.",
                model="model-123",
                llm_provider="openai",
            )
        )

        await self._instance(prisma, llm_router).check_batch_cost()

        prisma.db.litellm_managedobjecttable.update.assert_awaited_once()
        assert prisma.db.litellm_managedobjecttable.update.call_args[1]["data"] == {
            "batch_processed": True
        }

    @pytest.mark.asyncio
    async def test_provider_404_with_deployment_gone_keeps_job(self):
        """With the batch's own deployment removed from the router, default fallbacks can
        send the retrieve to a provider that never saw the batch. That 404 proves nothing,
        so the row must stay unprocessed instead of losing its spend forever."""
        import litellm

        prisma = self._prisma(
            [
                self._job(
                    "job-misrouted",
                    self._encode("litellm_proxy;model_id:model-gone;llm_batch_id:batch_alive"),
                )
            ]
        )
        llm_router = MagicMock()
        llm_router.get_deployment = MagicMock(return_value=None)
        llm_router.aretrieve_batch = AsyncMock(
            side_effect=litellm.NotFoundError(
                message="No batch found with id 'batch_alive'.",
                model="model-gone",
                llm_provider="openai",
            )
        )

        await self._instance(prisma, llm_router).check_batch_cost()

        prisma.db.litellm_managedobjecttable.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_provider_error_keeps_job_for_retry(self):
        """A failure that may clear up (timeout, 5xx) must still leave the row unprocessed."""
        prisma = self._prisma(
            [
                self._job(
                    "job-flaky",
                    self._encode("litellm_proxy;model_id:model-123;llm_batch_id:batch_flaky"),
                )
            ]
        )
        llm_router = MagicMock()
        llm_router.aretrieve_batch = AsyncMock(side_effect=Exception("connection reset"))

        await self._instance(prisma, llm_router).check_batch_cost()

        prisma.db.litellm_managedobjecttable.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retirement_falls_back_to_status_without_batch_processed_column(self):
        """Older schemas have no batch_processed column, so the only way to stop selecting
        the row is the status filter the poll query already applies."""
        prisma = self._prisma(
            [self._job("job-legacy", self._encode("litellm_proxy;llm_batch_id:poison-no-model"))]
        )
        instance = self._instance(prisma, MagicMock())
        instance._has_batch_processed_column = False

        await instance.check_batch_cost()

        assert prisma.db.litellm_managedobjecttable.update.call_args[1]["data"] == {
            "status": "stale_expired"
        }

    @pytest.mark.asyncio
    async def test_stale_cleanup_gives_up_on_never_costed_completed_rows(self):
        """A row already in a terminal status is never rewritten by the staleness sweep, so
        it needs its own bound or it starves newer batches indefinitely."""
        prisma = self._prisma([])

        await self._instance(prisma, MagicMock()).check_batch_cost()

        calls = prisma.db.litellm_managedobjecttable.update_many.call_args_list
        assert len(calls) == 2, "expected the staleness sweep plus the never-costed sweep"
        where = calls[1][1]["where"]
        assert where["file_purpose"] == "batch"
        assert where["batch_processed"] is False
        assert where["status"] == {"in": ["complete", "completed"]}
        assert "created_at" in where
        assert calls[1][1]["data"] == {"batch_processed": True}

    @pytest.mark.asyncio
    async def test_newer_batch_is_polled_once_dead_rows_are_retired(self):
        """The end state the customer cares about: dead rows retire on the cycle they are
        first seen, and the healthy batch behind them keeps getting polled."""
        dead_rows = [
            self._job("job-no-model", self._encode("litellm_proxy;llm_batch_id:poison-no-model")),
            self._job(
                "job-gone",
                self._encode("litellm_proxy;model_id:model-123;llm_batch_id:batch_deadbeef"),
            ),
        ]
        live_row = self._job(
            "job-live",
            self._encode("litellm_proxy;model_id:model-123;llm_batch_id:batch_live"),
        )
        prisma = self._prisma(dead_rows + [live_row])

        import litellm

        in_progress = MagicMock()
        in_progress.status = "in_progress"

        async def _retrieve(model, batch_id, litellm_metadata):
            if batch_id == "batch_deadbeef":
                raise litellm.NotFoundError(
                    message=f"No batch found with id '{batch_id}'.",
                    model=model,
                    llm_provider="openai",
                )
            return in_progress

        llm_router = MagicMock()
        llm_router.aretrieve_batch = AsyncMock(side_effect=_retrieve)

        await self._instance(prisma, llm_router).check_batch_cost()

        retired = [
            call[1]["where"]["id"]
            for call in prisma.db.litellm_managedobjecttable.update.call_args_list
        ]
        assert retired == ["job-no-model", "job-gone"]
        assert (
            llm_router.aretrieve_batch.await_args_list[-1][1]["batch_id"] == "batch_live"
        ), "the newer healthy batch must still be polled in the same cycle"

    @pytest.mark.asyncio
    async def test_404_that_does_not_name_the_batch_keeps_job_for_retry(self):
        """A 404 about something other than the batch, e.g. a renamed Azure deployment, is
        fixable in config, so the row must survive to be costed after the fix."""
        import litellm

        prisma = self._prisma(
            [
                self._job(
                    "job-bad-deployment",
                    self._encode("litellm_proxy;model_id:model-123;llm_batch_id:batch_real"),
                )
            ]
        )
        llm_router = MagicMock()
        llm_router.aretrieve_batch = AsyncMock(
            side_effect=litellm.NotFoundError(
                message="Error code: 404 - DeploymentNotFound",
                model="model-123",
                llm_provider="azure",
            )
        )

        await self._instance(prisma, llm_router).check_batch_cost()

        prisma.db.litellm_managedobjecttable.update.assert_not_awaited()

class _FakeManagedObjectRow:
    """One managed batch row the provider has finished but nothing has costed yet."""

    def __init__(self):
        self.id = "job-claim-1"
        self.unified_object_id = _CLAIM_UNIFIED_BATCH_ID
        self.model_object_id = "batch-456"
        self.file_purpose = "batch"
        self.status = "in_progress"
        self.batch_processed = False
        self.created_by = "user-1"
        self.team_id = None
        self.api_key = None
        self.request_tags = None
        self.created_at = 1700000000
        self.file_object = json.dumps(
            {"id": "batch-456", "status": "in_progress", "input_file_id": "file-input-1",
             "output_file_id": _CLAIM_OUTPUT_FILE_ID}
        )


class _FakeManagedObjectTable:
    """A LiteLLM_ManagedObjectTable double backed by one real, mutable row.

    It honours the batch_processed and status filters, so the poller's compare-and-swap
    and the managed-files deletion guard both read the same state a shared Postgres row
    would give them. Staleness sweeps (the only queries scoped by created_at) never match.
    """

    def __init__(self, row: _FakeManagedObjectRow, journal: list):
        self.row = row
        self.journal = journal
        self.update_many = AsyncMock(side_effect=self._update_many)
        self.update = AsyncMock(side_effect=self._update)
        self.find_many = AsyncMock(side_effect=self._find_many)
        self.find_first = AsyncMock(return_value=None)

    def _matches(self, where: dict) -> bool:
        for key, value in where.items():
            if key == "created_at":
                return False
            if key == "status":
                if self.row.status in value.get("not_in", []):
                    return False
                if "in" in value and self.row.status not in value["in"]:
                    return False
            elif getattr(self.row, key) != value:
                return False
        return True

    async def _update_many(self, *, where: dict, data: dict) -> int:
        if not self._matches(where):
            return 0
        if "batch_processed" in where:
            self.journal.append("claim" if data.get("batch_processed") else "release")
        for key, value in data.items():
            setattr(self.row, key, value)
        return 1

    async def _update(self, *, where: dict, data: dict) -> None:
        self.journal.append("finalize")
        for key, value in data.items():
            setattr(self.row, key, value)

    async def _find_many(self, *, where: dict, take=None, order=None) -> list:
        return [self.row] if self._matches(where) else []


class TestMultiPodBatchCostClaim:
    """LIT-4827 regression: every pod and uvicorn worker schedules its own poller against
    the shared LiteLLM_ManagedObjectTable, so a completed batch must be claimed atomically
    before its cost is logged. Without the claim two pods select the same row in one window
    and both write an aretrieve_batch spend log for it, double counting the spend.

    The claim sits immediately before the spend-log write rather than before the results
    fetch, because batch_processed is also what keeps an unbilled row selectable by later
    poll cycles and what blocks deletion of the files the fetch reads."""

    @staticmethod
    def _instance(prisma, llm_router):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.get_proxy_hook.return_value = None
        return CheckBatchCost(
            proxy_logging_obj=proxy_logging_obj,
            prisma_client=prisma,
            llm_router=llm_router,
        )

    @staticmethod
    def _prisma(row: _FakeManagedObjectRow, journal: list):
        prisma = MagicMock()
        prisma.db.litellm_managedobjecttable = _FakeManagedObjectTable(row, journal)
        prisma.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
        prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
        return prisma

    @staticmethod
    def _router():
        response = MagicMock()
        response.status = "completed"
        response.output_file_id = _CLAIM_OUTPUT_FILE_ID
        response.error_file_id = None
        response.created_at = 1
        response.completed_at = 2
        response.model_dump_json.return_value = '{"id":"batch-456","status":"completed"}'

        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "openai"
        deployment.litellm_params.model = "gpt-4"
        deployment.model_info.model_dump.return_value = {}

        router = MagicMock()
        router.aretrieve_batch = AsyncMock(return_value=response)
        router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )
        router.get_deployment = MagicMock(return_value=deployment)
        return router

    @staticmethod
    @contextmanager
    def _billing_patches(journal: list, during_fetch=None, bill_error=None):
        """Patch the cost path a batch runs through, journalling the results fetch and the
        spend-log write. during_fetch runs while the output file is being read, which is
        the window an interrupted worker or a concurrent file deletion lands in."""
        file_content = MagicMock()
        file_content.content = b'{"id":"req-1"}'

        async def _afile_content(**kwargs):
            journal.append("fetch")
            if during_fetch is not None:
                await during_fetch()
            return file_content

        async def _bill(**kwargs):
            journal.append("bill")
            if bill_error is not None:
                raise bill_error

        def _is_b64(file_id):
            if file_id == _CLAIM_UNIFIED_BATCH_ID:
                return "llm_model_id,model-123;llm_batch_id,batch-456;"
            return False

        logging_obj = MagicMock()
        logging_obj.async_success_handler = AsyncMock(side_effect=_bill)

        with (
            patch(_IS_B64, side_effect=_is_b64),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_model_id_from_unified_batch_id",
                return_value="model-123",
            ),
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils.get_batch_id_from_unified_batch_id",
                return_value="batch-456",
            ),
            patch("litellm.files.main.afile_content", new=AsyncMock(side_effect=_afile_content)),
            patch(
                "litellm.batches.batch_utils._get_file_content_as_dictionary",
                return_value=[{"id": "req-1"}],
            ),
            patch(
                "litellm.batches.batch_utils.calculate_batch_cost_and_usage",
                new_callable=AsyncMock,
                return_value=(0.01, {"prompt_tokens": 10, "completion_tokens": 5}, ["gpt-4"]),
            ),
            patch(
                "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
                return_value=("gpt-4", "openai", None, None),
            ),
            patch("litellm.litellm_core_utils.litellm_logging.Logging", return_value=logging_obj),
        ):
            yield logging_obj

    @staticmethod
    def _claim_calls(prisma) -> list:
        return [
            call.kwargs
            for call in prisma.db.litellm_managedobjecttable.update_many.call_args_list
            if "id" in call.kwargs["where"]
        ]

    @staticmethod
    async def _run_deletion_guard(prisma, file_id: str) -> None:
        """Run the real managed-files deletion guard against the row the poller is costing."""
        from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

        cache = MagicMock()
        cache.async_get_cache = AsyncMock(return_value=None)
        cache.async_set_cache = AsyncMock()
        guard = _PROXY_LiteLLMManagedFiles(internal_usage_cache=cache, prisma_client=prisma)

        scheduler = MagicMock()
        scheduler.get_job.return_value = MagicMock()
        with patch("litellm.proxy.proxy_server.scheduler", scheduler):
            await guard._check_file_deletion_allowed(file_id)

    @pytest.mark.asyncio
    async def test_winning_pod_claims_the_row_between_fetching_and_billing(self):
        """The claim flips batch_processed false -> true after the results are in hand and
        before the spend log is written, so a concurrent pod's claim finds no matching row."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)

        with self._billing_patches(journal) as logging_obj:
            await self._instance(prisma, self._router()).check_batch_cost()

        assert journal == ["fetch", "claim", "bill", "finalize"]
        assert self._claim_calls(prisma) == [
            {
                "where": {"id": "job-claim-1", "batch_processed": False},
                "data": {"batch_processed": True},
            }
        ]
        logging_obj.async_success_handler.assert_awaited_once()
        assert row.batch_processed is True

    @pytest.mark.asyncio
    async def test_a_pod_that_loses_the_claim_after_fetching_does_not_bill(self):
        """Both pods select the row and fetch its results in the same window. The one whose
        compare-and-swap finds the row already taken must not write a second spend log."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)

        async def _other_pod_wins_the_row():
            row.batch_processed = True

        with self._billing_patches(journal, during_fetch=_other_pod_wins_the_row) as logging_obj:
            await self._instance(prisma, self._router()).check_batch_cost()

        assert journal == ["fetch"]
        logging_obj.async_success_handler.assert_not_awaited()
        assert self._claim_calls(prisma) == [
            {
                "where": {"id": "job-claim-1", "batch_processed": False},
                "data": {"batch_processed": True},
            }
        ]

    @pytest.mark.asyncio
    async def test_a_failed_spend_log_write_releases_the_claim(self):
        """A transient failure while billing a claimed batch must hand the row back, or its
        spend is silently lost instead of being retried on the next cycle."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)

        with self._billing_patches(journal, bill_error=Exception("spend log write failed")):
            await self._instance(prisma, self._router()).check_batch_cost()

        assert journal == ["fetch", "claim", "bill", "release"]
        assert row.batch_processed is False
        assert self._claim_calls(prisma)[-1] == {
            "where": {"id": "job-claim-1", "batch_processed": True},
            "data": {"batch_processed": False},
        }

    @pytest.mark.asyncio
    async def test_a_worker_interrupted_mid_costing_leaves_the_batch_billable(self):
        """A pod killed while reading a batch's results must leave the row for a later
        cycle. Claiming before the fetch marked the batch processed for good, so the pod
        that died took that batch's spend with it and no other pod ever selected it."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)
        reached_fetch = asyncio.Event()

        async def _never_returns():
            reached_fetch.set()
            await asyncio.Event().wait()

        with self._billing_patches(journal, during_fetch=_never_returns) as logging_obj:
            interrupted = asyncio.create_task(
                self._instance(prisma, self._router()).check_batch_cost()
            )
            await asyncio.wait_for(reached_fetch.wait(), timeout=5)
            assert row.batch_processed is False, "an in-flight costing must not mark the row processed"
            interrupted.cancel()
            with pytest.raises(asyncio.CancelledError):
                await interrupted

        assert journal == ["fetch"]
        logging_obj.async_success_handler.assert_not_awaited()

        survivor_journal = []
        survivor_prisma = self._prisma(row, survivor_journal)
        with self._billing_patches(survivor_journal) as survivor_logging:
            await self._instance(survivor_prisma, self._router()).check_batch_cost()

        assert survivor_journal == ["fetch", "claim", "bill", "finalize"]
        survivor_logging.async_success_handler.assert_awaited_once()
        assert row.batch_processed is True

    @pytest.mark.asyncio
    async def test_costing_in_flight_keeps_the_referenced_file_undeletable(self):
        """The deletion guard only holds files whose batch still has batch_processed false,
        so claiming the row before the fetch let a concurrent delete remove the very output
        file the in-flight costing was about to read."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)
        reached_fetch = asyncio.Event()
        finish_fetch = asyncio.Event()

        async def _wait_for_the_delete_attempt():
            reached_fetch.set()
            await finish_fetch.wait()

        with self._billing_patches(journal, during_fetch=_wait_for_the_delete_attempt):
            costing = asyncio.create_task(
                self._instance(prisma, self._router()).check_batch_cost()
            )
            await asyncio.wait_for(reached_fetch.wait(), timeout=5)

            with pytest.raises(HTTPException) as blocked:
                await self._run_deletion_guard(prisma, _CLAIM_OUTPUT_FILE_ID)
            assert blocked.value.status_code == 400
            assert _CLAIM_OUTPUT_FILE_ID in blocked.value.detail

            finish_fetch.set()
            await asyncio.wait_for(costing, timeout=5)

        assert journal == ["fetch", "claim", "bill", "finalize"]
        assert row.batch_processed is True
        await self._run_deletion_guard(prisma, _CLAIM_OUTPUT_FILE_ID)

    @pytest.mark.asyncio
    async def test_schema_without_batch_processed_still_bills(self):
        """Older schemas have no column to claim, so they keep the pre-fix behavior instead
        of losing every batch's cost."""
        row = _FakeManagedObjectRow()
        journal = []
        prisma = self._prisma(row, journal)
        instance = self._instance(prisma, self._router())
        instance._has_batch_processed_column = False

        with self._billing_patches(journal) as logging_obj:
            await instance.check_batch_cost()

        assert self._claim_calls(prisma) == []
        assert journal == ["fetch", "bill", "finalize"]
        logging_obj.async_success_handler.assert_awaited_once()
