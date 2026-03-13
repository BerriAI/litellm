"""
Unit tests for memory leak fixes.

Tests verify:
1. Logging._cleanup_after_logging() properly clears large data
2. spend_log_transactions queue stays bounded
3. Reduced deepcopy in spend tracking
4. Periodic memory cleanup utility
"""

import asyncio
import copy
import datetime
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLoggingCleanupAfterLogging:
    """Tests for Logging._cleanup_after_logging()"""

    def _make_logging_obj(self) -> Any:
        """Create a minimal Logging object with populated fields."""
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="test-model",
            messages=[{"role": "user", "content": "x" * 10000}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )

        # Populate fields that should be cleaned up
        logging_obj.model_call_details["httpx_response"] = "large_response_object"
        logging_obj.model_call_details["original_response"] = "x" * 50000
        logging_obj.model_call_details["input"] = [
            {"role": "user", "content": "x" * 10000}
        ]
        logging_obj.model_call_details["additional_args"] = {
            "complete_input_dict": {"messages": [{"role": "user", "content": "x" * 10000}]}
        }
        logging_obj.model_call_details["standard_logging_object"] = {"large": "payload"}
        logging_obj.model_call_details[
            "complete_streaming_response"
        ] = "streaming_response"
        logging_obj.model_call_details[
            "async_complete_streaming_response"
        ] = "async_streaming_response"
        logging_obj.model_call_details[
            "raw_request_typed_dict"
        ] = {"body": "x" * 10000}

        # Populate streaming chunks
        logging_obj.streaming_chunks = [{"chunk": i} for i in range(100)]
        logging_obj.sync_streaming_chunks = [{"chunk": i} for i in range(100)]

        return logging_obj

    def test_should_clear_model_call_details_keys(self):
        """_cleanup_after_logging should remove all large keys from model_call_details."""
        logging_obj = self._make_logging_obj()

        # Verify fields are populated before cleanup
        assert "httpx_response" in logging_obj.model_call_details
        assert "original_response" in logging_obj.model_call_details
        assert "input" in logging_obj.model_call_details
        assert "additional_args" in logging_obj.model_call_details
        assert "standard_logging_object" in logging_obj.model_call_details

        # Run cleanup
        logging_obj._cleanup_after_logging()

        # Verify all large keys are removed
        assert "httpx_response" not in logging_obj.model_call_details
        assert "original_response" not in logging_obj.model_call_details
        assert "input" not in logging_obj.model_call_details
        assert "additional_args" not in logging_obj.model_call_details
        assert "standard_logging_object" not in logging_obj.model_call_details
        assert "complete_streaming_response" not in logging_obj.model_call_details
        assert "async_complete_streaming_response" not in logging_obj.model_call_details
        assert "raw_request_typed_dict" not in logging_obj.model_call_details

    def test_should_clear_streaming_chunks(self):
        """_cleanup_after_logging should clear streaming_chunks and sync_streaming_chunks."""
        logging_obj = self._make_logging_obj()

        assert len(logging_obj.streaming_chunks) == 100
        assert len(logging_obj.sync_streaming_chunks) == 100

        logging_obj._cleanup_after_logging()

        assert len(logging_obj.streaming_chunks) == 0
        assert len(logging_obj.sync_streaming_chunks) == 0

    def test_should_clear_messages(self):
        """_cleanup_after_logging should set messages to None."""
        logging_obj = self._make_logging_obj()

        assert logging_obj.messages is not None

        logging_obj._cleanup_after_logging()

        assert logging_obj.messages is None

    def test_should_preserve_non_large_model_call_details(self):
        """_cleanup_after_logging should NOT remove non-large fields."""
        logging_obj = self._make_logging_obj()

        # These should survive cleanup
        logging_obj.model_call_details["model"] = "test-model"
        logging_obj.model_call_details["litellm_call_id"] = "test-id"

        logging_obj._cleanup_after_logging()

        assert logging_obj.model_call_details["model"] == "test-model"
        assert logging_obj.model_call_details["litellm_call_id"] == "test-id"

    def test_should_be_idempotent(self):
        """_cleanup_after_logging should be safe to call multiple times."""
        logging_obj = self._make_logging_obj()

        logging_obj._cleanup_after_logging()
        # Should not raise
        logging_obj._cleanup_after_logging()
        logging_obj._cleanup_after_logging()


class TestSpendLogQueueBounding:
    """Tests for bounded spend_log_transactions queue."""

    @pytest.mark.asyncio
    async def test_should_bound_spend_log_queue(self):
        """Queue should not exceed MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE."""
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()

        # Create a mock PrismaClient
        mock_prisma = MagicMock()
        mock_prisma.spend_log_transactions = []
        mock_prisma._spend_log_transactions_lock = asyncio.Lock()

        max_size = 100  # Use a small value for testing
        with patch(
            "litellm.constants.MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE",
            max_size,
        ):
            # Fill the queue beyond the limit
            for i in range(max_size + 50):
                await writer._insert_spend_log_to_db(
                    payload={"request_id": f"req-{i}", "spend": 0.01},
                    prisma_client=mock_prisma,
                )

        # Queue should be bounded: after exceeding limit, oldest 10% are dropped
        # then the new item is appended.
        # The first time it exceeds, it drops 10 (10% of 100) then appends, = 91
        # This continues for each subsequent insert...
        # Final size should be <= max_size + 1 (at most one over limit before next trim)
        assert len(mock_prisma.spend_log_transactions) <= max_size + 1

    @pytest.mark.asyncio
    async def test_should_drop_oldest_entries(self):
        """When queue is full, oldest entries should be dropped."""
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()

        mock_prisma = MagicMock()
        mock_prisma.spend_log_transactions = []
        mock_prisma._spend_log_transactions_lock = asyncio.Lock()

        max_size = 20
        with patch(
            "litellm.constants.MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE",
            max_size,
        ):
            # Fill exactly to the limit
            for i in range(max_size):
                await writer._insert_spend_log_to_db(
                    payload={"request_id": f"req-{i}", "spend": 0.01},
                    prisma_client=mock_prisma,
                )

            assert len(mock_prisma.spend_log_transactions) == max_size

            # Add one more - should trigger drop of oldest 10% (2 items)
            await writer._insert_spend_log_to_db(
                payload={"request_id": "req-new", "spend": 0.01},
                prisma_client=mock_prisma,
            )

            # After drop: 20 - 2 = 18, then +1 = 19
            assert len(mock_prisma.spend_log_transactions) == max_size - (max_size // 10) + 1

            # The newest entry should be at the end
            assert mock_prisma.spend_log_transactions[-1]["request_id"] == "req-new"

            # The oldest entries (req-0, req-1) should be gone
            remaining_ids = [
                entry["request_id"] for entry in mock_prisma.spend_log_transactions
            ]
            assert "req-0" not in remaining_ids
            assert "req-1" not in remaining_ids


class TestDeepCopyReduction:
    """Tests for reduced deepcopy in spend tracking."""

    @pytest.mark.asyncio
    async def test_should_work_with_spend_logs_disabled(self):
        """When disable_spend_logs=True, no deepcopy should occur."""
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()

        original_payload = {
            "request_id": "test-123",
            "spend": 0.05,
            "startTime": "2026-03-13T00:00:00",
            "endTime": "2026-03-13T00:00:01",
            "model": "gpt-4",
            "api_key": "hashed-key",
            "user": "user-123",
            "team_id": "",
            "request_tags": ["tag1", "tag2"],
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        deepcopy_count = 0
        original_deepcopy = copy.deepcopy

        def counting_deepcopy(obj, memo=None):
            nonlocal deepcopy_count
            deepcopy_count += 1
            return original_deepcopy(obj, memo)

        with patch("litellm.proxy.db.db_spend_update_writer.copy.deepcopy", counting_deepcopy):
            with patch(
                "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter._batch_database_updates",
                new_callable=AsyncMock,
            ):
                with patch("litellm.proxy.proxy_server.disable_spend_logs", True):
                    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
                        with patch(
                            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock()
                        ):
                            with patch(
                                "litellm.proxy.proxy_server.litellm_proxy_budget_name",
                                None,
                            ):
                                with patch(
                                    "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
                                    return_value=original_payload,
                                ):
                                    with patch(
                                        "litellm.proxy.utils.ProxyUpdateSpend.disable_spend_updates",
                                        return_value=False,
                                    ):
                                        await writer.update_database(
                                            token="sk-test",
                                            user_id="user-123",
                                            end_user_id=None,
                                            team_id=None,
                                            org_id=None,
                                            kwargs={},
                                            completion_response=None,
                                            start_time=datetime.datetime.now(),
                                            end_time=datetime.datetime.now(),
                                            response_cost=0.05,
                                        )

        # When spend logs are disabled, there should be 0 deepcopy calls
        assert deepcopy_count == 0, (
            f"Expected 0 deepcopy calls when spend logs disabled, got {deepcopy_count}"
        )


class TestPeriodicMemoryCleanup:
    """Tests for the periodic memory cleanup utility."""

    def test_should_run_gc_collect(self):
        """_periodic_memory_cleanup should call gc.collect()."""
        from litellm.proxy.common_utils.memory_utils import _periodic_memory_cleanup

        with patch("litellm.proxy.common_utils.memory_utils.gc") as mock_gc:
            mock_gc.collect.return_value = 42
            _periodic_memory_cleanup()
            mock_gc.collect.assert_called_once()

    def test_should_call_malloc_trim_on_linux(self):
        """_periodic_memory_cleanup should call malloc_trim on Linux."""
        from litellm.proxy.common_utils import memory_utils

        # Mock the module-level variables
        mock_libc = MagicMock()
        original_libc = memory_utils._libc
        original_available = memory_utils._malloc_trim_available

        try:
            memory_utils._libc = mock_libc
            memory_utils._malloc_trim_available = True

            memory_utils._periodic_memory_cleanup()

            mock_libc.malloc_trim.assert_called_once_with(0)
        finally:
            memory_utils._libc = original_libc
            memory_utils._malloc_trim_available = original_available

    def test_should_handle_malloc_trim_failure_gracefully(self):
        """_periodic_memory_cleanup should not raise if malloc_trim fails."""
        from litellm.proxy.common_utils import memory_utils

        mock_libc = MagicMock()
        mock_libc.malloc_trim.side_effect = OSError("test error")
        original_libc = memory_utils._libc
        original_available = memory_utils._malloc_trim_available

        try:
            memory_utils._libc = mock_libc
            memory_utils._malloc_trim_available = True

            # Should not raise
            memory_utils._periodic_memory_cleanup()
        finally:
            memory_utils._libc = original_libc
            memory_utils._malloc_trim_available = original_available
