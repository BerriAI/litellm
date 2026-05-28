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
    async def test_raw_output_file_id_converted_to_managed_id(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """CheckBatchCost must register managed unified IDs for raw provider
        output_file_id / error_file_id when no managed_file row already exists,
        using the batch creator as the auth context.
        """
        import base64
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )
        # No existing managed_file rows — resolve_* finds nothing, so the helper
        # registers new managed IDs for output/error.
        mock_prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(
            return_value=None
        )

        unified_object_id_raw = (
            "litellm_proxy;model_id:model-123;llm_batch_id:batch-456"
        )
        unified_object_id = (
            base64.urlsafe_b64encode(unified_object_id_raw.encode())
            .decode()
            .rstrip("=")
        )

        mock_job = MagicMock()
        mock_job.id = "job-raw-file-1"
        mock_job.unified_object_id = unified_object_id
        mock_job.created_by = "user-1"
        mock_job.team_id = None

        check_batch_cost_instance._has_batch_processed_column = True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        raw_output_file_id = "file-batch-output-abc123"
        raw_error_file_id = "file-batch-error-xyz456"
        fake_managed_output_id = "bGl0ZWxsbV9wcm94eTo6bw=="
        fake_managed_error_id = "bGl0ZWxsbV9wcm94eTo6ZQ=="

        from litellm.types.utils import LiteLLMBatch
        mock_response = LiteLLMBatch(
            id="batch-456",
            completion_window="24h",
            created_at=1700000000,
            endpoint="/v1/chat/completions",
            input_file_id="file-batch-input-raw",
            object="batch",
            status="completed",
            completed_at=1700001000,
            output_file_id=raw_output_file_id,
            error_file_id=raw_error_file_id,
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

        with (
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
        stored = {
            next(iter(c[1]["model_mappings"].values())): c[1]["file_id"]
            for c in mock_hook.store_unified_file_id.call_args_list
        }
        assert stored == {
            raw_output_file_id: fake_managed_output_id,
            raw_error_file_id: fake_managed_error_id,
        }
        for c in mock_hook.store_unified_file_id.call_args_list:
            assert c[1]["user_api_key_dict"].user_id == "user-1"
        assert mock_response.output_file_id == fake_managed_output_id
        assert mock_response.error_file_id == fake_managed_error_id

    @pytest.mark.asyncio
    async def test_input_file_id_resolved_to_existing_managed_id(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """LIT-3386: when the raw input_file_id has a managed_file row (the user
        uploaded the file through the proxy and got back a unified ID),
        CheckBatchCost must resolve it to that managed ID before persisting the
        batch's file_object — otherwise GET /v1/batches/{id} returns the raw
        provider ID for input_file_id and the user cannot correlate it back to
        their managed upload.

        The retrieve/cancel HTTP path already does this via
        ensure_batch_response_managed_file_ids; CheckBatchCost must too.
        """
        import base64, json
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        raw_input_file_id = "file-batch-input-raw-aaa"
        raw_output_file_id = "file-batch-output-raw-bbb"
        managed_input_unified_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29ubDt1bmlmaWVkX2lkLGFiYy0xMjM7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00"  # base64-encoded form actually stored in DB

        async def _find_first(where, **kw):
            ids = (where or {}).get("flat_model_file_ids", {})
            if isinstance(ids, dict) and ids.get("has") == raw_input_file_id:
                row = MagicMock()
                row.unified_file_id = managed_input_unified_id
                return row
            return None

        mock_prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(
            side_effect=_find_first
        )

        unified_object_id_raw = (
            "litellm_proxy;model_id:model-x;llm_batch_id:batch-y"
        )
        unified_object_id = (
            base64.urlsafe_b64encode(unified_object_id_raw.encode())
            .decode()
            .rstrip("=")
        )

        mock_job = MagicMock()
        mock_job.id = "job-input-resolve-1"
        mock_job.unified_object_id = unified_object_id
        mock_job.created_by = "shin-real-user"
        mock_job.team_id = "team-1"

        check_batch_cost_instance._has_batch_processed_column = True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        from litellm.types.utils import LiteLLMBatch
        response = LiteLLMBatch(
            id="batch-y",
            completion_window="24h",
            created_at=1700000000,
            endpoint="/v1/chat/completions",
            input_file_id=raw_input_file_id,
            object="batch",
            status="completed",
            completed_at=1700001000,
            output_file_id=raw_output_file_id,
        )
        mock_llm_router.aretrieve_batch = AsyncMock(return_value=response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )
        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "openai"
        deployment.litellm_params.model = "gpt-4"
        deployment.model_name = "gpt-4-batch"
        deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=deployment)

        hook = MagicMock()
        hook.get_unified_output_file_id.return_value = "bGl0ZWxsbV9wcm94eTo6bw=="
        hook.store_unified_file_id = AsyncMock()
        check_batch_cost_instance.proxy_logging_obj.get_proxy_hook.return_value = hook

        file_content = MagicMock()
        file_content.content = b'{"id":"req-1"}'

        with (
            patch("litellm.files.main.afile_content", new_callable=AsyncMock, return_value=file_content),
            patch("litellm.batches.batch_utils._get_file_content_as_dictionary", return_value=[{"id": "req-1"}]),
            patch("litellm.batches.batch_utils.calculate_batch_cost_and_usage", new_callable=AsyncMock, return_value=(0.01, {"prompt_tokens": 1, "completion_tokens": 1}, ["gpt-4"])),
            patch("litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider", return_value=("gpt-4", "openai", None, None)),
            patch("litellm.litellm_core_utils.litellm_logging.Logging") as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj
            await check_batch_cost_instance.check_batch_cost()

        # The persisted file_object must carry the managed input_file_id, not the raw provider value.
        assert mock_prisma_client.db.litellm_managedobjecttable.update.await_count == 1
        update_data = mock_prisma_client.db.litellm_managedobjecttable.update.call_args[1]["data"]
        persisted = json.loads(update_data["file_object"])
        assert persisted["input_file_id"] == managed_input_unified_id, (
            f"expected managed unified id, got {persisted}"
        )

    @pytest.mark.asyncio
    async def test_existing_output_managed_row_is_reused_not_duplicated(
        self, check_batch_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """LIT-3386: when a managed_file row already exists for the raw
        output_file_id (e.g. the retrieve_batch HTTP path ran first),
        CheckBatchCost must reuse that row instead of registering a new one
        (which would double-write).
        """
        import base64
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
        )
        mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=None
        )

        raw_output_file_id = "file-output-raw-xxx"
        existing_managed_output_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29ubDt1bmlmaWVkX2lkLG91dC0xO3RhcmdldF9tb2RlbF9uYW1lcyxncHQtNA"  # base64-encoded form actually stored in DB

        async def _find_first(where, **kw):
            ids = (where or {}).get("flat_model_file_ids", {})
            if isinstance(ids, dict) and ids.get("has") == raw_output_file_id:
                row = MagicMock()
                row.unified_file_id = existing_managed_output_id
                return row
            return None

        mock_prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(
            side_effect=_find_first
        )

        unified_object_id_raw = "litellm_proxy;model_id:m-1;llm_batch_id:b-1"
        unified_object_id = base64.urlsafe_b64encode(
            unified_object_id_raw.encode()
        ).decode().rstrip("=")

        mock_job = MagicMock()
        mock_job.id = "job-dedup-1"
        mock_job.unified_object_id = unified_object_id
        mock_job.created_by = "user-1"
        mock_job.team_id = None
        check_batch_cost_instance._has_batch_processed_column = True
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        from litellm.types.utils import LiteLLMBatch
        response = LiteLLMBatch(
            id="b-1",
            completion_window="24h",
            created_at=1700000000,
            endpoint="/v1/chat/completions",
            input_file_id="file-input-irrelevant",
            object="batch",
            status="completed",
            completed_at=1700001000,
            output_file_id=raw_output_file_id,
        )
        mock_llm_router.aretrieve_batch = AsyncMock(return_value=response)
        mock_llm_router.get_deployment_credentials_with_provider = MagicMock(
            return_value={"api_key": "sk-test"}
        )
        deployment = MagicMock()
        deployment.litellm_params.custom_llm_provider = "openai"
        deployment.litellm_params.model = "gpt-4"
        deployment.model_name = "gpt-4-batch"
        deployment.model_info.model_dump.return_value = {}
        mock_llm_router.get_deployment = MagicMock(return_value=deployment)

        hook = MagicMock()
        hook.get_unified_output_file_id.return_value = "SHOULD-NOT-BE-CALLED"
        hook.store_unified_file_id = AsyncMock()
        check_batch_cost_instance.proxy_logging_obj.get_proxy_hook.return_value = hook

        file_content = MagicMock()
        file_content.content = b'{"id":"req-1"}'

        with (
            patch("litellm.files.main.afile_content", new_callable=AsyncMock, return_value=file_content),
            patch("litellm.batches.batch_utils._get_file_content_as_dictionary", return_value=[{"id": "req-1"}]),
            patch("litellm.batches.batch_utils.calculate_batch_cost_and_usage", new_callable=AsyncMock, return_value=(0.01, {"prompt_tokens": 1, "completion_tokens": 1}, ["gpt-4"])),
            patch("litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider", return_value=("gpt-4", "openai", None, None)),
            patch("litellm.litellm_core_utils.litellm_logging.Logging") as mock_logging_cls,
        ):
            mock_logging_obj = MagicMock()
            mock_logging_obj.async_success_handler = AsyncMock()
            mock_logging_cls.return_value = mock_logging_obj
            await check_batch_cost_instance.check_batch_cost()

        # Existing managed row was reused: response.output_file_id is the existing managed id,
        # and we did NOT call store_unified_file_id / get_unified_output_file_id for output.
        assert response.output_file_id == existing_managed_output_id
        assert hook.store_unified_file_id.await_count == 0
        assert hook.get_unified_output_file_id.call_count == 0
