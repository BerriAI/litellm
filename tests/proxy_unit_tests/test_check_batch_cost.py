"""
Unit tests for CheckBatchCost class.
Covers: stale-row cleanup (file_purpose scoping), paginated find_many,
the batch_processed-column fallback query, and routing of unmanaged
Vertex (raw gs:// input_file_id) and Bedrock (raw s3:// input_file_id,
ARN unified_object_id) batches with no managed unified id.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_IS_B64 = "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id"


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
            return_value=0
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
    async def test_find_many_uses_pagination_and_excludes_stale(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """find_many is called with take, order, and all terminal statuses excluded."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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

    @pytest.mark.asyncio
    async def test_fallback_query_used_when_batch_processed_missing(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """Falls back to query without batch_processed when primary query raises."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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

    @pytest.mark.asyncio
    async def test_column_absence_cached_across_cycles(
        self, check_batch_cost_instance, mock_prisma_client
    ):
        """After column absence is discovered, subsequent cycles skip the primary query entirely."""
        from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
            return_value=0
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
    async def test_primary_path_completion_update_includes_batch_processed(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """When batch_processed column IS present, completion update must set it to True.

        This is the symmetric counterpart to test_fallback_completion_update_omits_batch_processed
        and proves the conditional on _has_batch_processed_column governs the update data.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
            return_value=0
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
        """When the provider reports a terminal status (failed/expired/cancelled), the row
        must be written back with that status and batch_processed=True so it stops being
        polled forever.
        """
        from unittest.mock import patch

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        mock_job = MagicMock()
        mock_job.id = "job-terminal-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="
        mock_job.created_by = "user-1"

        assert check_batch_cost_instance._has_batch_processed_column is True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_response = MagicMock()
        mock_response.status = terminal_status
        mock_response.model_dump_json.return_value = (
            f'{{"id":"batch-1","status":"{terminal_status}"}}'
        )

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
    async def test_raw_output_file_id_converted_to_managed_id(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """CheckBatchCost must convert a raw provider output_file_id to a managed base64 ID.

        Without this, GET /batches/{id} returns a raw file ID that cannot be routed
        through the proxy, causing API_KEY errors when clients call GET /files/{id}/content.
        """
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
            model_name="gpt-5-mini",
        )
        mock_hook.get_unified_output_file_id.assert_any_call(
            output_file_id=raw_error_file_id,
            model_id="model-123",
            model_name="gpt-5-mini",
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
        prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=0)
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
        prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=0)
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


class TestBatchCostAttribution:
    """A batch-cost spend log must be attributed to the creating key/team/tags the same
    way a non-batch request is. Regression for batches created by keys with no user_id,
    where the row was previously dropped and identity was blank."""

    def _instance(self):
        from litellm_enterprise.proxy.common_utils.check_batch_cost import (
            CheckBatchCost,
        )

        instance = CheckBatchCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=MagicMock(),
            llm_router=MagicMock(),
        )
        instance.prisma_client.db = MagicMock()
        return instance

    @pytest.mark.asyncio
    async def test_get_user_info_skips_query_when_user_id_none(self):
        """find_unique(where={"user_id": None}) raises "A value is required but not set";
        the None user_id case (team/service-account keys) must short-circuit instead."""
        instance = self._instance()
        instance.prisma_client.db.litellm_usertable.find_unique = AsyncMock()

        result = await instance._get_user_info("batch-1", None)

        assert result == {}
        instance.prisma_client.db.litellm_usertable.find_unique.assert_not_called()

    @pytest.mark.asyncio
    async def test_attribution_metadata_replays_key_team_and_tags(self):
        from types import SimpleNamespace

        instance = self._instance()
        instance.prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=SimpleNamespace(user_email="creator@example.test")
        )
        instance.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
            return_value=SimpleNamespace(key_alias="prod-batch-key")
        )
        instance.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=SimpleNamespace(team_alias="growth-team")
        )
        job = SimpleNamespace(
            created_by="user-1",
            team_id="team-1",
            user_api_key="hashed-key-abc",
            request_tags=["env:prod", "team:growth"],
        )

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["user_api_key"] == "hashed-key-abc"
        assert metadata["user_api_key_user_id"] == "user-1"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert metadata["user_api_key_alias"] == "prod-batch-key"
        assert metadata["user_api_key_team_alias"] == "growth-team"
        assert metadata["user_api_key_user_email"] == "creator@example.test"
        assert metadata["tags"] == ["env:prod", "team:growth"]

    @pytest.mark.asyncio
    async def test_attribution_metadata_keeps_key_when_no_user_id(self):
        """A team/service-account key has no created_by. The persisted key hash and
        team_id must still flow through so _should_track_cost_callback writes the row."""
        from types import SimpleNamespace

        from litellm.proxy.hooks.proxy_track_cost_callback import (
            _should_track_cost_callback,
        )

        instance = self._instance()
        instance.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
            return_value=None
        )
        instance.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=None
        )
        job = SimpleNamespace(
            created_by=None,
            team_id="team-1",
            user_api_key="hashed-key-abc",
            request_tags=None,
        )

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["user_api_key"] == "hashed-key-abc"
        assert metadata["user_api_key_user_id"] is None
        assert "tags" not in metadata
        assert (
            _should_track_cost_callback(
                user_api_key=metadata["user_api_key"],
                user_id=metadata.get("user_api_key_user_id"),
                team_id=metadata.get("user_api_key_team_id"),
                end_user_id=None,
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_attribution_metadata_tolerates_legacy_rows(self):
        """Older managed-object rows predate user_api_key/request_tags. Attribution must
        still fall back to created_by/team_id without raising."""
        from types import SimpleNamespace

        instance = self._instance()
        instance.prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )
        instance.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=None
        )
        job = SimpleNamespace(created_by="user-1", team_id="team-1")

        metadata = await instance._build_creator_attribution_metadata(job, "batch-1")

        assert metadata["user_api_key"] is None
        assert metadata["user_api_key_user_id"] == "user-1"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert "tags" not in metadata
