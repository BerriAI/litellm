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
        # First find_many (primary query) raises; second (fallback) returns empty list
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            side_effect=[Exception("column batch_processed does not exist"), []]
        )

        await check_batch_cost_instance.check_batch_cost()

        calls = mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args_list
        assert len(calls) == 2
        fallback_where = calls[1][1]["where"]
        # Fallback must not reference batch_processed
        assert "batch_processed" not in fallback_where
        # Fallback must exclude stale_expired
        assert "stale_expired" in fallback_where["status"]["not_in"]
        # Fallback must still paginate
        assert calls[1][1]["take"] == MAX_OBJECTS_PER_POLL_CYCLE
