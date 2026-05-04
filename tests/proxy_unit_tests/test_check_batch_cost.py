"""
Unit tests for CheckBatchCost class.
Covers: stale-row cleanup (file_purpose scoping), paginated find_many,
and the batch_processed-column fallback query.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
        return MagicMock()

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
