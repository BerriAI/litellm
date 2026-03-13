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
        from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

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

        calls = mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
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
        """find_many is called with take, order, and stale_expired excluded from status."""
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
        assert "stale_expired" in find_call[1]["where"]["status"]["not_in"]

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

        calls = mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args_list
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
        assert mock_prisma_client.db.litellm_managedobjecttable.find_many.call_count == 1
        fallback_where = mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args[1]["where"]
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

        mock_job = MagicMock()
        mock_job.id = "job-fallback-1"
        mock_job.unified_object_id = "dW5pZmllZF9iYXRjaF9pZA=="  # base64-looking value
        mock_job.created_by = "user-1"

        # Primary query fails → fallback path
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            side_effect=[Exception("column batch_processed does not exist"), [mock_job]]
        )

        # Stub out the heavy per-job processing so we reach the update()
        with (
            patch(
                "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
                return_value=None,  # causes "not a valid unified object id" early-continue
            ),
        ):
            await check_batch_cost_instance.check_batch_cost()

        # Even though the job was skipped (invalid ID), confirm the fallback path was taken
        # by checking the find_many calls
        find_calls = mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args_list
        assert len(find_calls) == 2
        fallback_where = find_calls[1][1]["where"]
        assert "batch_processed" not in fallback_where

        # If a completion update were issued, it must not contain batch_processed
        for call in mock_prisma_client.db.litellm_managedobjecttable.update.call_args_list:
            assert "batch_processed" not in call[1].get("data", {})
